from neo4j import GraphDatabase

d = GraphDatabase.driver('neo4j://127.0.0.1:7687', auth=('neo4j', '12345678'))

with d.session() as s:
    r = s.run('MATCH (rec:Record) WHERE rec.ho_ten_norm IS NOT NULL RETURN count(rec) AS cnt').data()
    print('Records with ho_ten_norm:', r[0]['cnt'])

    r2 = s.run('MATCH (rec:Record) WHERE rec.ho_ten_norm = $n RETURN rec.raw_data, rec.file_nguon LIMIT 3', n='luu van phuc').data()
    print('Luu Van Phuc records:', len(r2))
    for row in r2:
        print(' ', str(row['rec.raw_data'])[:200])
        print('  file_nguon:', row['rec.file_nguon'])

    # Test CONTAINS search
    r3 = s.run('MATCH (rec:Record)-[:THUOC_TB]->(ann:Announcement) WHERE rec.ho_ten_norm CONTAINS $n RETURN ann.title, rec.file_nguon, rec.ho_ten_norm LIMIT 5', n='luu van phuc').data()
    print('\nCONTAINS search result:')
    for row in r3: print(' ', row)

d.close()

