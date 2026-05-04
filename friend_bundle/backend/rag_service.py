from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
import json
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchAny, MatchValue
from sentence_transformers import CrossEncoder, SentenceTransformer


from backend.config import (
    EMBED_MODEL1,
    EMBED_MODEL2,
    ENABLE_NEO4J,
    ENABLE_RERANKER,
    GENERATION_BACKOFF_SECONDS,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    MAX_CTX_CHARS,
    MAX_GENERATION_RETRIES,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    QDRANT_COLLECTION,
    QDRANT_ADDITIONAL_COLLECTION,
    QDRANT_PATH,
    RERANK_CANDIDATES,
    RERANK_TOP_K,
    RERANKER_LOCAL_ONLY,
    RERANKER_MODEL,
    TOP_K,
)

try:
    from groq import Groq
except Exception:  # pragma: no cover
    Groq = None

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None


SYSTEM_PROMPT = """Ban la tro ly thong tin cua Truong Dai hoc Bach khoa, Dai hoc Da Nang (DUT).
Chi tra loi dua tren ngu canh duoc cung cap.
Neu thong tin khong du de ket luan, hay noi ro la chua tim thay du lieu du.
Tra loi bang tieng Viet, ngan gon, chinh xac, co the neu nguon bang ann_id hoac tieu de khi phu hop."""

ROOM_RE = re.compile(r"\b(?:phong|p\.?)\s*([a-z]{1,3}\d{0,3})\b", re.I)
MSSV_RE = re.compile(r"\b(10\d{6,10})\b")
DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
ANN_RE = re.compile(r"\bANN_[0-9A-F]{10}\b", re.I)
MONTH_YEAR_RE = re.compile(r"\b(?:thang|tháng)?\s*(\d{1,2})[/-](20\d{2})\b", re.I)
COHORT_RE = re.compile(r"\b(?:khoa|khóa|k)\s*[-_ ]?(20\d{2}|\d{2})\b", re.I)
NAME_RE = re.compile(r"\b([A-ZĐ][a-zà-ỹ]+(?:\s+[A-ZĐ][a-zà-ỹ]+){1,4})\b")
STOP_NAME_PHRASES = {
    "Trường Đại Học",
    "Đại Học Đà",
    "Đại Học Bách",
    "Phòng Đào Tạo",
    "Phòng Công Tác",
    "Đợt Thi",
    "Kỳ Thi",
    "Môn Công",
    "Môn Vật",
}


@dataclass
class QueryEntities:
    mssv: str | None = None
    student_name: str | None = None
    room: str | None = None
    date_text: str | None = None
    ann_id: str | None = None
    exam_type: str | None = None
    batch_no: str | None = None
    batch_text: str | None = None
    month_text: str | None = None
    cohort: str | None = None
    subject: str | None = None
    intents: list[str] = field(default_factory=list)
    raw_keywords: list[str] = field(default_factory=list)


@dataclass
class QueryPlan:
    route: str
    intents: list[str]
    graph_mode: str | None
    vector_enabled: bool
    reason: str


@dataclass
class GraphHint:
    mode: str
    ann_ids: list[str]
    graph_records: list[dict[str, Any]]
    summary: dict[str, Any]


class RagService:
    def __init__(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.embedder1 = SentenceTransformer(EMBED_MODEL1, local_files_only=True)  # 768d → dut_chunks
        self.embedder2 = SentenceTransformer(EMBED_MODEL2, local_files_only=True)
        self.reranker = None
        self.embedder = self.embedder1

        if ENABLE_RERANKER:
            self.reranker = CrossEncoder(
                RERANKER_MODEL,
                local_files_only=RERANKER_LOCAL_ONLY,
            )
        self.qdrant = QdrantClient(path=str(QDRANT_PATH))
        self.llm_provider = LLM_PROVIDER
        # self.ollama_base_url = OLLAMA_BASE_URL.rstrip("/")
        self.groq = Groq(api_key=GROQ_API_KEY) if (Groq and GROQ_API_KEY) else None
        self.neo4j_driver = None

        if ENABLE_NEO4J and GraphDatabase and NEO4J_PASSWORD:
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
            )
            self.neo4j_driver.verify_connectivity()

    def close(self) -> None:
        if self.neo4j_driver is not None:
            self.neo4j_driver.close()
        try:
            self.qdrant.close()
        except Exception as e:
            print("Error closing Qdrant client:", e)
            pass

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return re.sub(r"\s+", " ", text).strip()

    def normalize_cohort(self, raw: str) -> str:
        if len(raw) == 2:
            return f"20{raw}"
        return raw

    def extract_student_name(self, query: str) -> str | None:
        explicit = re.search(
            r"(?:Sinh viên|SV)\s+([A-ZĐ][a-zà-ỹ]+(?:\s+[A-ZĐ][a-zà-ỹ]+){1,5})",
            query,
        )
        if explicit:
            name = explicit.group(1).strip()
            if name not in STOP_NAME_PHRASES:
                return name

        candidates = []
        for match in NAME_RE.finditer(query):
            name = match.group(1).strip()
            if name in STOP_NAME_PHRASES:
                continue
            if len(name.split()) < 2:
                continue
            candidates.append(name)
        return max(candidates, key=len) if candidates else None

    def extract_subject(self, query: str, q_norm: str) -> str | None:
        patterns = [
            r"\bmon\s+([^?.,;\n]{3,80})",
            r"\bhoc phan\s+([^?.,;\n]{3,80})",
        ]
        for pat in patterns:
            m = re.search(pat, q_norm)
            if m:
                subject = m.group(1).strip(" -:")
                subject = re.split(
                    r"\b(?:o|trong|vao|ngay|dot|khoa|lop|khong|the nao|la gi|bao nhieu|co ai)\b",
                    subject,
                    maxsplit=1,
                )[0].strip()
                if len(subject) >= 3:
                    return subject
        return None

    def extract_entities(self, query: str) -> QueryEntities:
        q_norm = self.normalize_text(query)
        entities = QueryEntities()

        mssv_match = MSSV_RE.search(query)
        if mssv_match:
            entities.mssv = mssv_match.group(1)

        room_match = ROOM_RE.search(q_norm)
        if room_match:
            entities.room = room_match.group(1).upper()

        date_match = DATE_RE.search(query)
        if date_match:
            entities.date_text = date_match.group(1)

        month_match = MONTH_YEAR_RE.search(q_norm)
        if month_match:
            entities.month_text = f"{int(month_match.group(1))}/{month_match.group(2)}"

        ann_match = ANN_RE.search(query.upper())
        if ann_match:
            entities.ann_id = ann_match.group(0).upper()

        if "toeic" in q_norm:
            entities.exam_type = "TOEIC"
        elif "toefl" in q_norm:
            entities.exam_type = "TOEFL"
        elif "vstep" in q_norm:
            entities.exam_type = "VSTEP"
        elif "gdqp" in q_norm or "quoc phong" in q_norm:
            entities.exam_type = "GDQP"

        batch_match = re.search(r"\bdot\s*(\d+)\b", q_norm)
        if batch_match:
            entities.batch_no = batch_match.group(1)
            entities.batch_text = f"dot {entities.batch_no}"

        cohort_match = COHORT_RE.search(q_norm)
        if cohort_match:
            entities.cohort = self.normalize_cohort(cohort_match.group(1))

        entities.student_name = self.extract_student_name(query)
        entities.subject = self.extract_subject(query, q_norm)

        if any(k in q_norm for k in ["co ai", "nhung ai", "danh sach", "bao nhieu nguoi", "tham gia"]):
            entities.intents.append("participant_lookup")
        if any(k in q_norm for k in ["phong", "phong thi", "ca thi"]):
            entities.intents.append("room_lookup")
        if entities.mssv:
            entities.intents.append("student_lookup")
        if entities.student_name and not entities.mssv and not entities.subject:
            entities.intents.append("student_name_lookup")
        if entities.subject:
            entities.intents.append("subject_lookup")
        if entities.cohort:
            entities.intents.append("cohort_lookup")
        if entities.batch_no:
            entities.intents.append("batch_lookup")
        if any(k in q_norm for k in ["quy dinh", "thu tuc", "huong dan", "yeu cau"]):
            entities.intents.append("policy_lookup")
        if any(k in q_norm for k in ["thong bao", "ann_", "van ban"]):
            entities.intents.append("announcement_lookup")

        for keyword in [
            "toeic", "toefl", "vstep", "gdqp", "tot nghiep", "tuyen",
            "thi", "phong", "mon", "khoa", "dot",
        ]:
            if keyword in q_norm:
                entities.raw_keywords.append(keyword)

        seen = set()
        entities.intents = [x for x in entities.intents if not (x in seen or seen.add(x))]
        return entities

    def plan_retrieval(self, query: str, entities: QueryEntities) -> QueryPlan:
        if self.neo4j_driver is None:
            return QueryPlan(
                route="vector_only",
                intents=entities.intents,
                graph_mode=None,
                vector_enabled=True,
                reason="Neo4j chưa bật nên chỉ dùng vector retrieval.",
            )

        if entities.ann_id:
            return QueryPlan(
                route="hybrid",
                intents=entities.intents,
                graph_mode="announcement",
                vector_enabled=True,
                reason="Có ann_id cụ thể nên filter graph/vector theo thông báo.",
            )

        if entities.mssv:
            return QueryPlan(
                route="hybrid",
                intents=entities.intents or ["student_lookup"],
                graph_mode="student",
                vector_enabled=True,
                reason="Có MSSV nên dùng graph trước, rồi vector lấy ngữ cảnh văn bản.",
            )

        if entities.student_name and any(
            intent in entities.intents for intent in ["student_name_lookup", "participant_lookup"]
        ):
            return QueryPlan(
                route="hybrid",
                intents=entities.intents,
                graph_mode="student_name",
                vector_enabled=True,
                reason="Có tên sinh viên nên tra graph theo ho_ten trước, rồi vector theo ann_id.",
            )

        if any([entities.room, entities.subject, entities.cohort, entities.batch_no, entities.date_text, entities.exam_type, entities.month_text]) and (
            any(
                intent in entities.intents
                for intent in ["participant_lookup", "room_lookup", "subject_lookup", "cohort_lookup", "batch_lookup"]
            )
            or bool(entities.date_text or entities.month_text or entities.exam_type)
        ):
            return QueryPlan(
                route="hybrid",
                intents=entities.intents,
                graph_mode="exam_search",
                vector_enabled=True,
                reason="Có thực thể cấu trúc (phòng/môn/khóa/đợt) nên graph lọc bản ghi thi trước.",
            )

        if "policy_lookup" in entities.intents:
            return QueryPlan(
                route="vector_only",
                intents=entities.intents,
                graph_mode=None,
                vector_enabled=True,
                reason="Câu hỏi quy định chung phù hợp semantic search hơn graph.",
            )

        return QueryPlan(
            route="vector_only",
            intents=entities.intents,
            graph_mode=None,
            vector_enabled=True,
            reason="Không có thực thể cấu trúc mạnh nên dùng vector retrieval mặc định.",
        )

    def graph_lookup(self, entities: QueryEntities, plan: QueryPlan) -> GraphHint | None:
        if self.neo4j_driver is None or plan.graph_mode is None:
            return None
        if plan.graph_mode == "student" and entities.mssv:
            return self.graph_lookup_student(entities.mssv)
        if plan.graph_mode == "student_name" and entities.student_name:
            return self.graph_lookup_student_name(entities)
        if plan.graph_mode == "exam_search":
            return self.graph_lookup_exam_entities(entities)
        if plan.graph_mode == "announcement" and entities.ann_id:
            return self.graph_lookup_announcement(entities.ann_id)
        return None

    def graph_lookup_student(self, mssv: str) -> GraphHint | None:
        cypher = """
        MATCH (sv:SinhVien {mssv: $mssv})
        OPTIONAL MATCH (sv)-[:THAM_GIA_THI]->(ki:KiThi)-[:THUOC_TB]->(ann1:Announcement)
        WITH sv, collect(DISTINCT {
            type: 'exam',
            ann_id: ann1.id,
            title: ann1.title,
            loai_thi: ki.loai_thi,
            ngay_thi: ki.ngay_thi,
            gio_thi: ki.gio_thi,
            phong_thi: ki.phong_thi
        }) AS exams
        OPTIONAL MATCH (sv)-[:CO_TRONG_TB]->(ann2:Announcement)
        RETURN
            sv.mssv AS mssv,
            sv.ho_ten AS ho_ten,
            exams AS exams,
            collect(DISTINCT {
                type: 'announcement',
                ann_id: ann2.id,
                title: ann2.title,
                file_nguon: ann2.file_nguon
            }) AS announcements
        """
        with self.neo4j_driver.session() as session:
            record = session.run(cypher, mssv=mssv).single()
        if not record:
            return None
        exams = [x for x in (record["exams"] or []) if x.get("ann_id")]
        anns = [x for x in (record["announcements"] or []) if x.get("ann_id")]
        ann_ids = sorted({x["ann_id"] for x in exams + anns if x.get("ann_id")})
        return GraphHint(
            mode="student",
            ann_ids=ann_ids,
            graph_records=exams + anns,
            summary={"mssv": record["mssv"], "student_name": record.get("ho_ten")},
        )

    def graph_lookup_student_name(self, entities: QueryEntities) -> GraphHint | None:
        name = entities.student_name or ""
        cypher = """
        MATCH (sv:SinhVien)
        WHERE toLower(coalesce(sv.ho_ten, '')) CONTAINS toLower($name)
        OPTIONAL MATCH (sv)-[:THAM_GIA_THI]->(ki:KiThi)-[:THUOC_TB]->(ann:Announcement)
        RETURN sv.mssv AS mssv,
               sv.ho_ten AS ho_ten,
               collect(DISTINCT {
                   type: 'exam',
                   ann_id: ann.id,
                   title: ann.title,
                   loai_thi: ki.loai_thi,
                   ngay_thi: ki.ngay_thi,
                   gio_thi: ki.gio_thi,
                   phong_thi: ki.phong_thi
               })[0..20] AS exams
        LIMIT 10
        """
        with self.neo4j_driver.session() as session:
            rows = list(session.run(cypher, name=name))
        if not rows:
            return None
        records = []
        ann_ids = []
        for row in rows:
            exam_records = [x for x in (row["exams"] or []) if x.get("ann_id")]
            ann_ids.extend(x["ann_id"] for x in exam_records)
            records.append(
                {
                    "type": "student_name_match",
                    "mssv": row["mssv"],
                    "ho_ten": row["ho_ten"],
                    "exams": exam_records,
                }
            )
        return GraphHint(
            mode="student_name",
            ann_ids=sorted(set(ann_ids)),
            graph_records=records,
            summary={"student_name": name, "matches": len(records)},
        )

    def graph_lookup_exam_entities(self, entities: QueryEntities) -> GraphHint | None:
        params: dict[str, Any] = {
            "room": entities.room or "",
            "date_text": entities.date_text or "",
            "exam_type": entities.exam_type or "",
            "batch_text": entities.batch_text or "",
            "month_text": entities.month_text or "",
            "cohort": entities.cohort or "",
            "subject": entities.subject or "",
        }
        cypher = """
        MATCH (sv:SinhVien)-[:THAM_GIA_THI]->(ki:KiThi)-[:THUOC_TB]->(ann:Announcement)
        WHERE ($room = '' OR toUpper(coalesce(ki.phong_thi, '')) CONTAINS toUpper($room))
          AND ($date_text = '' OR coalesce(ki.ngay_thi, '') CONTAINS $date_text)
          AND ($exam_type = '' OR toUpper(coalesce(ki.loai_thi, '')) CONTAINS toUpper($exam_type))
          AND ($batch_text = '' OR toLower(coalesce(ann.title, '')) CONTAINS toLower($batch_text))
          AND ($month_text = '' OR toLower(coalesce(ann.title, '')) CONTAINS toLower($month_text) OR coalesce(ki.ngay_thi, '') CONTAINS $month_text)
          AND ($cohort = '' OR toLower(coalesce(ann.title, '')) CONTAINS toLower($cohort) OR toLower(coalesce(ki.title, '')) CONTAINS toLower($cohort))
          AND ($subject = '' OR toLower(coalesce(ann.title, '')) CONTAINS toLower($subject) OR toLower(coalesce(ki.title, '')) CONTAINS toLower($subject))
        RETURN ann.id AS ann_id,
               ann.title AS title,
               ki.loai_thi AS loai_thi,
               ki.ngay_thi AS ngay_thi,
               ki.gio_thi AS gio_thi,
               ki.phong_thi AS phong_thi,
               count(DISTINCT sv) AS student_count,
               collect(DISTINCT {mssv: sv.mssv, ho_ten: sv.ho_ten})[0..25] AS students
        ORDER BY ngay_thi, gio_thi
        LIMIT 12
        """
        with self.neo4j_driver.session() as session:
            rows = list(session.run(cypher, **params))
        if not rows:
            return None
        records = []
        ann_ids = []
        for row in rows:
            ann_id = row["ann_id"]
            if ann_id:
                ann_ids.append(ann_id)
            records.append(
                {
                    "type": "exam_search",
                    "ann_id": ann_id,
                    "title": row["title"],
                    "loai_thi": row["loai_thi"],
                    "ngay_thi": row["ngay_thi"],
                    "gio_thi": row["gio_thi"],
                    "phong_thi": row["phong_thi"],
                    "student_count": row["student_count"],
                    "students": row["students"],
                }
            )
        return GraphHint(
            mode="exam_search",
            ann_ids=sorted(set(ann_ids)),
            graph_records=records,
            summary={
                "room": entities.room,
                "date_text": entities.date_text,
                "exam_type": entities.exam_type,
                "batch_text": entities.batch_text,
                "month_text": entities.month_text,
                "cohort": entities.cohort,
                "subject": entities.subject,
            },
        )

    def graph_lookup_announcement(self, ann_id: str) -> GraphHint | None:
        cypher = """
        MATCH (ann:Announcement {id: $ann_id})
        OPTIONAL MATCH (ki:KiThi)-[:THUOC_TB]->(ann)
        RETURN ann.id AS ann_id,
               ann.title AS title,
               ann.file_nguon AS file_nguon,
               collect(DISTINCT {
                   loai_thi: ki.loai_thi,
                   ngay_thi: ki.ngay_thi,
                   gio_thi: ki.gio_thi,
                   phong_thi: ki.phong_thi
               })[0..20] AS exams
        """
        with self.neo4j_driver.session() as session:
            row = session.run(cypher, ann_id=ann_id).single()
        if not row:
            return None
        return GraphHint(
            mode="announcement",
            ann_ids=[ann_id],
            graph_records=[{
                "type": "announcement",
                "ann_id": row["ann_id"],
                "title": row["title"],
                "file_nguon": row["file_nguon"],
                "exams": row["exams"],
            }],
            summary={"ann_id": ann_id, "title": row["title"]},
        )

    def build_vector_filter(
        self,
        entities: QueryEntities,
        ann_ids: list[str] | None = None,
        doc_type_hint: str | None = None,
    ) -> Filter | None:
        must = []
        if ann_ids:
            must.append(FieldCondition(key="ann_id", match=MatchAny(any=ann_ids)))
        elif entities.ann_id:
            must.append(FieldCondition(key="ann_id", match=MatchValue(value=entities.ann_id)))

        if doc_type_hint:
            must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type_hint)))

        if entities.exam_type and not doc_type_hint:
            dt_map = {
                "TOEIC": "tb_dang_ky_thi",
                "TOEFL": "tb_dang_ky_thi",
                "VSTEP": "tb_dang_ky_thi",
                "GDQP": "gdqp",
            }
            doc_type = dt_map.get(entities.exam_type)
            if doc_type:
                must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))

        return Filter(must=must) if must else None

    # def retrieve_vector(
    #     self,
    #     query: str,
    #     top_k: int = TOP_K,
    #     entities: QueryEntities | None = None,
    #     ann_ids: list[str] | None = None,
    #     doc_type_hint: str | None = None,
    #     candidate_limit: int | None = None,
    #     use_reranker: bool | None = None,
    # ) -> list[dict[str, Any]]:
    #     q_vec = self.embedder.encode(query, normalize_embeddings=True).tolist()
    #     use_reranker = self.reranker is not None if use_reranker is None else use_reranker
    #     query_filter = self.build_vector_filter(
    #         entities or QueryEntities(),
    #         ann_ids=ann_ids,
    #         doc_type_hint=doc_type_hint,
    #     )
    #     search_limit = candidate_limit or top_k
    #     if use_reranker:
    #         search_limit = max(search_limit, top_k, RERANK_CANDIDATES)
    #     results = self.qdrant.query_points(
    #         collection_name=QDRANT_COLLECTION,
    #         query=q_vec,
    #         query_filter=query_filter,
    #         limit=search_limit,
    #         with_payload=True,
    #     ).points

    #     if not results and query_filter is not None:
    #         fallback_entities = entities or QueryEntities()
    #         query_filter = self.build_vector_filter(
    #             fallback_entities,
    #             ann_ids=ann_ids,
    #             doc_type_hint=None,
    #         )
    #         results = self.qdrant.query_points(
    #             collection_name=QDRANT_COLLECTION,
    #             query=q_vec,
    #             query_filter=query_filter,
    #             limit=search_limit,
    #             with_payload=True,
    #         ).points

    #     hits = [{
    #         "score": float(r.score),
    #         "chunk_id": r.payload.get("chunk_id", ""),
    #         "text": r.payload.get("text", ""),
    #         "title": r.payload.get("title", ""),
    #         "doc_type": r.payload.get("doc_type", ""),
    #         "ann_id": r.payload.get("ann_id", ""),
    #     } for r in results]
    #     if use_reranker and hits:
    #         hits = self.rerank_hits(query=query, hits=hits, top_k=top_k)
    #     else:
    #         hits = hits[:top_k]
    #     return hits

 
    def retrieve_vector(self, query, top_k=TOP_K, entities=None,
                    ann_ids=None, doc_type_hint=None,
                    candidate_limit=None, use_reranker=None):

        q_vec1 = self.embedder1.encode(query, normalize_embeddings=True).tolist()
        q_vec2 = self.embedder2.encode(query, normalize_embeddings=True).tolist()

        use_reranker = self.reranker is not None if use_reranker is None else use_reranker
        search_limit = max(candidate_limit or top_k, RERANK_CANDIDATES if use_reranker else top_k)

        # Collection 1
        query_filter = self.build_vector_filter(
            entities or QueryEntities(), ann_ids=ann_ids, doc_type_hint=doc_type_hint
        )
        results1 = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=q_vec1,
            query_filter=query_filter,
            limit=search_limit, with_payload=True,
        ).points

        hits1 = [{
            "score":    float(r.score),
            "chunk_id": r.payload.get("chunk_id", ""),
            "text":     r.payload.get("text", ""),
            "title":    r.payload.get("title", ""),
            "doc_type": r.payload.get("doc_type", ""),
            "ann_id":   r.payload.get("ann_id", ""),
            "source":   "dut_chunks",
        } for r in results1]

        # Collection 2
        results2 = self.qdrant.query_points(
            collection_name=QDRANT_ADDITIONAL_COLLECTION,
            query=q_vec2,
            query_filter=None,
            limit=search_limit, with_payload=True,
        ).points

        hits2 = [{
            "score":    float(r.score),
            "chunk_id": r.payload.get("chunk_id", ""),
            "text":     r.payload.get("text", ""),
            "title":    (r.payload.get("metadata") or {}).get("so_hieu", "Văn bản pháp quy"),
            "doc_type": (r.payload.get("metadata") or {}).get("linh_vuc", "phap_quy"),
            "ann_id":   (r.payload.get("metadata") or {}).get("id", ""),
            "source":   "chatbot_documents",
        } for r in results2]

        # ── RRF thay vì sort theo score thô ──
        all_hits = self._reciprocal_rank_fusion([hits1, hits2], top_k=search_limit)

        if use_reranker and all_hits:
            return self.rerank_hits(query=query, hits=all_hits, top_k=top_k)
        return all_hits[:top_k]


    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[dict]],
        top_k: int,
        k: int = 60,          # hằng số RRF, 60 là chuẩn
        weights: list[float] | None = None,  # None = equal weight
    ) -> list[dict]:
        """
        Gộp nhiều ranked list bằng RRF.
        Score RRF = sum( weight_i / (k + rank_i) )
        """
        if weights is None:
            weights = [1.0] * len(ranked_lists)

        # chunk_id → accumulated RRF score + metadata
        rrf_scores: dict[str, float] = {}
        hit_map: dict[str, dict] = {}

        for ranked_list, weight in zip(ranked_lists, weights):
            for rank, hit in enumerate(ranked_list, start=1):
                key = f"{hit['source']}::{hit['chunk_id']}"
                rrf_scores[key] = rrf_scores.get(key, 0.0) + weight / (k + rank)
                if key not in hit_map:
                    hit_map[key] = hit

        # Sort theo RRF score
        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        result = []
        for key in sorted_keys[:top_k]:
            hit = dict(hit_map[key])
            hit["score"] = rrf_scores[key]   # ghi đè score bằng RRF score
            result.append(hit)

        return result

        
    def rerank_hits(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self.reranker is None or len(hits) <= 1:
            return hits[:top_k]

        pairs = []
        for hit in hits:
            doc_text = hit["text"]
            if hit.get("title"):
                doc_text = f"{hit['title']}\n{doc_text}"
            pairs.append((query, doc_text))

        scores = self.reranker.predict(pairs)
        reranked: list[dict[str, Any]] = []
        for hit, score in zip(hits, scores, strict=False):
            item = dict(hit)
            item["retrieval_score"] = float(hit["score"])
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: (
                item.get("rerank_score", float("-inf")),
                item.get("retrieval_score", float("-inf")),
            ),
            reverse=True,
        )
        for rank, item in enumerate(reranked[:top_k], start=1):
            item["rank"] = rank
        return reranked[:top_k]

    def generate(
        self,
        query: str,
        context: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        max_retries: int = MAX_GENERATION_RETRIES,
    ) -> str:
        user_message = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}"
        # if self.llm_provider == "ollama":
        #     return self.generate_with_ollama(
        #         query=query,
        #         context=context,
        #         model=model or OLLAMA_MODEL,
        #         temperature=temperature,
        #         max_tokens=max_tokens,
        #     )
        if self.groq is None:
            return "Chua co GROQ_API_KEY nen backend moi tra ve context, chua sinh cau tra loi."

        attempt = 0
        while True:
            try:
                response = self.groq.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                attempt += 1
                if attempt > max_retries or not self.is_rate_limit_error(exc):
                    raise
                sleep_seconds = self.extract_retry_after_seconds(exc)
                if sleep_seconds is None:
                    sleep_seconds = GENERATION_BACKOFF_SECONDS * attempt
                time.sleep(min(sleep_seconds, 90.0))

    # def generate_with_ollama(
    #     self,
    #     query: str,
    #     context: str,
    #     model: str,
    #     temperature: float,
    #     max_tokens: int,
    # ) -> str:
    #     payload = {
    #         "model": model or OLLAMA_MODEL,
    #         "messages": [
    #             {"role": "system", "content": SYSTEM_PROMPT},
    #             {"role": "user", "content": f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}"},
    #         ],
    #         "stream": False,
    #         "options": {
    #             "temperature": temperature,
    #             "num_predict": max_tokens,
    #         },
    #     }
    #     req = urlrequest.Request(
    #         url=f"{self.ollama_base_url}/api/chat",
    #         data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    #         headers={"Content-Type": "application/json"},
    #         method="POST",
    #     )
    #     try:
    #         with urlrequest.urlopen(req, timeout=600) as resp:
    #             raw = resp.read().decode("utf-8")
    #     except urlerror.HTTPError as exc:
    #         detail = exc.read().decode("utf-8", errors="replace")
    #         raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    #     except urlerror.URLError as exc:
    #         raise RuntimeError(f"Cannot reach Ollama at {self.ollama_base_url}: {exc.reason}") from exc

    #     try:
    #         data = json.loads(raw)
    #     except Exception as exc:
    #         raise RuntimeError(f"Invalid Ollama response: {raw[:500]}") from exc
    #     return (data.get("message") or {}).get("content", "").strip()

    def is_rate_limit_error(self, exc: Exception) -> bool:
        return "429" in str(exc) or "rate limit" in str(exc).lower()

    def extract_retry_after_seconds(self, exc: Exception) -> float | None:
        message = str(exc)
        for pattern in [
            r"try again in ([0-9.]+)s",
            r"Please try again in ([0-9.]+)s",
        ]:
            match = re.search(pattern, message, flags=re.I)
            if match:
                return max(float(match.group(1)), 1.0)
        return None

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        doc_type_hint: str | None = None,
        force_vector_only: bool = False,
        use_reranker: bool | None = None,
    ) -> dict[str, Any]:
        entities = self.extract_entities(query)
        plan = (
            QueryPlan(
                route="vector_only",
                intents=entities.intents,
                graph_mode=None,
                vector_enabled=True,
                reason="Force vector-only retrieval cho benchmark/doc_type hint.",
            )
            if force_vector_only
            else self.plan_retrieval(query, entities)
        )
        graph_hint = self.graph_lookup(entities, plan)

        vector_hits = []
        if plan.vector_enabled:
            vector_hits = self.retrieve_vector(
                query=query,
                top_k=top_k,
                entities=entities,
                ann_ids=graph_hint.ann_ids if graph_hint else None,
                doc_type_hint=doc_type_hint,
                use_reranker=use_reranker,
            )
            if graph_hint and not vector_hits:
                vector_hits = self.retrieve_vector(
                    query=query,
                    top_k=top_k,
                    entities=entities,
                    ann_ids=None,
                    doc_type_hint=doc_type_hint,
                    use_reranker=use_reranker,
                )

        return {
            "entities": entities,
            "plan": plan,
            "graph": graph_hint,
            "vector_hits": vector_hits,
        }

    def build_context(self, retrieval: dict[str, Any]) -> str:
        parts: list[str] = []
        total = 0

        graph_hint: GraphHint | None = retrieval["graph"]
        if graph_hint:
            graph_lines = [f"Graph mode: {graph_hint.mode}"]
            for key, value in graph_hint.summary.items():
                if value:
                    graph_lines.append(f"{key}: {value}")

            for item in graph_hint.graph_records[:8]:
                if item["type"] == "exam":
                    graph_lines.append(
                        "Thong tin graph: "
                        f"ann_id={item.get('ann_id', '')}, title={item.get('title', '')}, "
                        f"loai_thi={item.get('loai_thi', '')}, ngay_thi={item.get('ngay_thi', '')}, "
                        f"gio_thi={item.get('gio_thi', '')}, phong_thi={item.get('phong_thi', '')}"
                    )
                elif item["type"] in {"exam_search", "room_exam"}:
                    student_names = ", ".join(
                        f"{s.get('ho_ten', '')} ({s.get('mssv', '')})" for s in item.get("students", [])[:10]
                    )
                    graph_lines.append(
                        "Thong tin graph: "
                        f"ann_id={item.get('ann_id', '')}, title={item.get('title', '')}, "
                        f"loai_thi={item.get('loai_thi', '')}, ngay_thi={item.get('ngay_thi', '')}, "
                        f"gio_thi={item.get('gio_thi', '')}, phong_thi={item.get('phong_thi', '')}, "
                        f"so_sinh_vien={item.get('student_count', 0)}, danh_sach_mau={student_names}"
                    )
                elif item["type"] == "student_name_match":
                    exam_lines = "; ".join(
                        f"{x.get('ann_id', '')}:{x.get('loai_thi', '')}:{x.get('phong_thi', '')}"
                        for x in item.get("exams", [])[:8]
                    )
                    graph_lines.append(
                        f"Thong tin graph: mssv={item.get('mssv', '')}, ho_ten={item.get('ho_ten', '')}, exams={exam_lines}"
                    )
                else:
                    graph_lines.append(
                        "Thong tin graph: "
                        f"ann_id={item.get('ann_id', '')}, title={item.get('title', '')}, "
                        f"file_nguon={item.get('file_nguon', '')}"
                    )

            graph_block = "[GRAPH]\n" + "\n".join(graph_lines)
            parts.append(graph_block)
            total += len(graph_block)

        for hit in retrieval["vector_hits"]:
            block = f"[{hit['doc_type']} | {hit['title']} | {hit['ann_id']}]\n{hit['text']}"
            if total + len(block) > MAX_CTX_CHARS:
                break
            parts.append(block)
            total += len(block)

        return "\n\n---\n\n".join(parts)

    def answer_from_graph(self, retrieval: dict[str, Any]) -> str | None:
        entities: QueryEntities = retrieval["entities"]
        graph_hint: GraphHint | None = retrieval["graph"]
        if graph_hint is None:
            return None

        if "participant_lookup" in entities.intents and graph_hint.mode == "exam_search":
            lines = []
            for item in graph_hint.graph_records[:5]:
                students = item.get("students") or []
                if not students:
                    continue
                title = item.get("title", "Thông báo không rõ")
                loai_thi = item.get("loai_thi", "")
                ngay_thi = item.get("ngay_thi", "")
                phong_thi = item.get("phong_thi", "")
                header = f"{title} | {loai_thi} | {ngay_thi} | phòng {phong_thi}"
                names = [
                    f"{student.get('ho_ten', '').strip()} ({student.get('mssv', '').strip()})"
                    for student in students
                    if student.get("mssv")
                ]
                if not names:
                    continue
                preview = "\n".join(f"- {name}" for name in names[:15])
                suffix = f"\n... và thêm {len(students) - 15} sinh viên khác." if len(students) > 15 else ""
                lines.append(f"{header}\n{preview}{suffix}")
            if lines:
                return "Danh sách sinh viên tìm được từ dữ liệu graph:\n\n" + "\n\n".join(lines)

        return None

    def answer(
        self,
        query: str,
        top_k: int = TOP_K,
        doc_type_hint: str | None = None,
        force_vector_only: bool = False,
        model: str | None = None,
        use_reranker: bool | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        retrieval = self.retrieve(
            query=query,
            top_k=top_k,
            doc_type_hint=doc_type_hint,
            force_vector_only=force_vector_only,
            use_reranker=use_reranker,
        )
        context = self.build_context(retrieval=retrieval)
        # Determine selected model based on configured provider.
        # if model:
        #     selected_model = model
        # else:
        #     selected_model = OLLAMA_MODEL if self.llm_provider == "ollama" else GROQ_MODEL

        # if self.llm_provider == "groq" and isinstance(selected_model, str) and ":" in selected_model:
        #     print(f"Ignoring incompatible model '{selected_model}' for Groq provider; using {GROQ_MODEL} instead")
        selected_model = GROQ_MODEL
        # if self.llm_provider == "ollama" and isinstance(selected_model, str) and ":" not in selected_model:
        #     # Best-effort: Ollama models typically include ':' namespace; warn if plain Groq model passed.
        #     print(f"Using model '{selected_model}' for Ollama provider (ensure correct Ollama model name)")
        answer = self.answer_from_graph(retrieval)
        if answer is None:
            answer = self.generate(
                query=query,
                context=context,
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "graph_enabled": self.neo4j_driver is not None,
            "reranker_enabled": self.reranker is not None,
            "model": selected_model,
            "entities": asdict(retrieval["entities"]),
            "plan": asdict(retrieval["plan"]),
            "graph_match": None if retrieval["graph"] is None else asdict(retrieval["graph"]),
            "vector_hits": retrieval["vector_hits"],
        }