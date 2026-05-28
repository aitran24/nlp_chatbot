from neo4j import GraphDatabase
d = GraphDatabase.driver('neo4j://127.0.0.1:7687', auth=('neo4j','12345678'))
with d.session() as s:
    r = s.run(
        'MATCH (sv:SinhVien) WHERE sv.ho_ten CONTAINS $n OR coalesce(sv.ho_ten_norm,"") CONTAINS $n2 RETURN sv.mssv, sv.ho_ten, sv.ho_ten_norm LIMIT 5',
        n='Thanh Hùng', n2='thanh hung'
    ).data()
    print('SinhVien match:', r)
    r2 = s.run(
        'MATCH (rec:Record) WHERE rec.raw_data CONTAINS $n RETURN rec.raw_data LIMIT 1',
        n='Thanh H'
    ).data()
    if r2:
        print('Record raw_data sample:', r2[0]['rec.raw_data'][:300])
    else:
        print('No record match')
d.close()
