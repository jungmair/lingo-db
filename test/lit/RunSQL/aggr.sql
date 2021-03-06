--//RUN: run-sql %s %S/../../../resources/data/uni | FileCheck %s

--//CHECK: |                      s.matrnr  |                           sum  |                           min  |                           max  |                           avg  |                         count  |                         count  |
--//CHECK: ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--//CHECK: |                         25403  |                             2  |                             2  |                             2  |                          2.00  |                             1  |                             1  |
--//CHECK: |                         26120  |                             4  |                             4  |                             4  |                          4.00  |                             1  |                             1  |
--//CHECK: |                         27550  |                             8  |                             4  |                             4  |                          4.00  |                             2  |                             2  |
--//CHECK: |                         28106  |                            11  |                             2  |                             4  |                          2.75  |                             4  |                             4  |
--//CHECK: |                         29120  |                            10  |                             2  |                             4  |                          3.33  |                             3  |                             3  |
--//CHECK: |                         29555  |                             6  |                             2  |                             4  |                          3.00  |                             2  |                             2  |


select s.matrnr, sum(v.sws),min(v.sws),max(v.sws),avg(v.sws*1.0),count(*),count(v.sws)
from studenten s, hoeren h, vorlesungen v
where s.matrnr=h.matrnr and h.vorlnr=v.vorlnr
group by s.matrnr
order by s.matrnr
