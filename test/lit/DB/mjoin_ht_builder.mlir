 // RUN: run-mlir %s | FileCheck %s
 !entry_type=type tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>>
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(2)
//CHECK: int(2)
//CHECK: string("---------------")
//CHECK: int(0)
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(4)
//CHECK: int(4)
//CHECK: string("---------------")
//CHECK: int(0)
//CHECK: string("---------------")
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(2)
//CHECK: int(2)
//CHECK: string("---------------")
//CHECK: int(1)
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(4)
//CHECK: int(4)
//CHECK: string("---------------")
//CHECK: int(1)
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("---------------")
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(4)
//CHECK: int(4)
//CHECK: int(1)
//CHECK: string("---------------")
//CHECK: string("stra")
//CHECK: int(4)
//CHECK: int(2)
//CHECK: int(2)
//CHECK: int(1)
//CHECK: string("---------------")
//CHECK: string("strc")
//CHECK: int(3)
//CHECK: int(3)
//CHECK: int(3)
//CHECK: int(0)
//CHECK: string("---------------")
//CHECK: string("strd")
//CHECK: int(1)
//CHECK: int(1)
//CHECK: int(1)
//CHECK: int(0)
//CHECK: string("---------------")
 module {

	func.func @main () {
         %str_const = db.constant ( "---------------" ) :!db.string

         %str1=db.constant ( "stra" ) :!db.string
         %str2=db.constant ( "strb" ) :!db.string
         %str3=db.constant ( "strc" ) :!db.string
         %str4=db.constant ( "strd" ) :!db.string
         %int1=db.constant ( 4 ) : i32
         %int2=db.constant ( 2 ) : i32
         %int3=db.constant ( 3 ) : i32
         %int4=db.constant ( 1 ) : i32
         %zero=db.constant ( 0 ) : i32
         %one=db.constant ( 1 ) : i32
        %default_marker= arith.constant 0 : i64
        %key1 = util.pack %str1, %int1 : !db.string,i32 -> tuple<!db.string,i32>
        %key2 = util.pack %str1, %int1 : !db.string,i32 -> tuple<!db.string,i32>
        %key3 = util.pack %str3, %int3 : !db.string,i32 -> tuple<!db.string,i32>
        %key4 = util.pack %str4, %int4 : !db.string,i32 -> tuple<!db.string,i32>

        %val1 = util.pack %default_marker, %int1, %int1 : i64,i32,i32 -> tuple<i64,i32,i32>
        %val2 = util.pack %default_marker, %int2, %int2 : i64,i32,i32 -> tuple<i64,i32,i32>
        %val3 = util.pack %default_marker, %int3, %int3 : i64,i32,i32 -> tuple<i64,i32,i32>
        %val4 = util.pack %default_marker, %int4, %int4 : i64,i32,i32 -> tuple<i64,i32,i32>

        %ht= dsa.create_ds  !dsa.join_ht<tuple<!db.string,i32>,tuple<i64,i32,i32>>
        dsa.ht_insert %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i64,i32,i32>>, %key1 : tuple<!db.string,i32>, %val1 : tuple<i64,i32,i32>  hash: (%key : tuple<!db.string,i32>){
                                                                                                                                                                 %h = db.hash %key : tuple<!db.string,i32>
                                                                                                                                                                 dsa.yield %h : index
                                                                                                                                                              }
        dsa.ht_insert %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i64,i32,i32>>, %key2 : tuple<!db.string,i32>, %val2 : tuple<i64,i32,i32>  hash: (%key : tuple<!db.string,i32>){
                                                                                                                                                                 %h = db.hash %key : tuple<!db.string,i32>
                                                                                                                                                                 dsa.yield %h : index
                                                                                                                                                              }
        dsa.ht_insert %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i64,i32,i32>>, %key3 : tuple<!db.string,i32>, %val3 : tuple<i64,i32,i32>  hash: (%key : tuple<!db.string,i32>){
                                                                                                                                                                 %h = db.hash %key : tuple<!db.string,i32>
                                                                                                                                                                 dsa.yield %h : index
                                                                                                                                                              }
        dsa.ht_insert %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i64,i32,i32>>, %key4 : tuple<!db.string,i32>, %val4 : tuple<i64,i32,i32>  hash: (%key : tuple<!db.string,i32>){
                                                                                                                                                          %h = db.hash %key : tuple<!db.string,i32>
                                                                                                                                                          dsa.yield %h : index
                                                                                                                                                       }
        dsa.finalize %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i64,i32,i32>>
        %hash = db.hash %key1  : tuple<!db.string,i32>
        %matches = dsa.lookup %ht :  !dsa.join_ht<tuple<!db.string,i32>,tuple<i64,i32,i32>>, %hash  : index -> !dsa.iterable<tuple<tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>>,!util.ref<tuple<i64,i32,i32>>>,join_ht_mod_iterator>
       dsa.for %entry in %matches : !dsa.iterable<tuple<tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>>,!util.ref<tuple<i64,i32,i32>>>,join_ht_mod_iterator> {
           %tpl,%ptr = util.unpack %entry : tuple<tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>>,!util.ref<tuple<i64,i32,i32>>> -> tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>>,!util.ref<tuple<i64,i32,i32>>
            %key,%val = util.unpack %tpl : tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>> -> tuple<!db.string,i32>,tuple<i64,i32,i32>
            %k1,%k2 = util.unpack %key : tuple<!db.string,i32> -> !db.string,i32
            %m,%v1,%v2 = util.unpack %val : tuple<i64,i32,i32> -> i64, i32,i32
            db.runtime_call "DumpValue" (%k1) : (!db.string) -> ()
            db.runtime_call "DumpValue" (%k2) : (i32) -> ()
            db.runtime_call "DumpValue" (%v1) : (i32) -> ()
            db.runtime_call "DumpValue" (%v2) : (i32) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
            %one_i64 = arith.constant 1 : i64
            %marker_ptr=util.generic_memref_cast %ptr : !util.ref<tuple<i64,i32,i32>> -> !util.ref<i64>
            %marker=util.to_memref %marker_ptr : !util.ref<i64> -> memref<i64>
            %marker_val=memref.atomic_rmw "assign" %one_i64 , %marker[] : (i64, memref<i64>) -> i64
            db.runtime_call "DumpValue" (%marker_val) : (i64) -> ()
        }
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        dsa.for %entry in %matches : !dsa.iterable<tuple<tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>>,!util.ref<tuple<i64,i32,i32>>>,join_ht_mod_iterator> {
             %tpl,%ptr = util.unpack %entry : tuple<tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>>,!util.ref<tuple<i64,i32,i32>>> -> tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>>,!util.ref<tuple<i64,i32,i32>>
             %key,%val = util.unpack %tpl : tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>> -> tuple<!db.string,i32>,tuple<i64,i32,i32>
             %k1,%k2 = util.unpack %key : tuple<!db.string,i32> -> !db.string,i32
             %m,%v1,%v2 = util.unpack %val : tuple<i64,i32,i32> -> i64, i32,i32
             db.runtime_call "DumpValue" (%k1) : (!db.string) -> ()
             db.runtime_call "DumpValue" (%k2) : (i32) -> ()
             db.runtime_call "DumpValue" (%v1) : (i32) -> ()
             db.runtime_call "DumpValue" (%v2) : (i32) -> ()
             db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
             %one_i64 = arith.constant 1 : i64
             %marker_ptr=util.generic_memref_cast %ptr : !util.ref<tuple<i64,i32,i32>> -> !util.ref<i64>
             %marker=util.to_memref %marker_ptr : !util.ref<i64> -> memref<i64>
             %marker_val=memref.atomic_rmw "assign" %one_i64 , %marker[] : (i64, memref<i64>) -> i64
             db.runtime_call "DumpValue" (%marker_val) : (i64) -> ()
         }
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()


            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()

        dsa.for %entry in %ht : !dsa.join_ht<tuple<!db.string,i32>,tuple<i64,i32,i32>> {
            %key,%val = util.unpack %entry : tuple<tuple<!db.string,i32>,tuple<i64,i32,i32>> -> tuple<!db.string,i32>,tuple<i64,i32,i32>
            %k1,%k2 = util.unpack %key : tuple<!db.string,i32> -> !db.string,i32
            %marker,%v1,%v2 = util.unpack %val : tuple<i64,i32,i32> -> i64,i32,i32
            db.runtime_call "DumpValue" (%k1) : (!db.string) -> ()
            db.runtime_call "DumpValue" (%k2) : (i32) -> ()
            db.runtime_call "DumpValue" (%v1) : (i32) -> ()
            db.runtime_call "DumpValue" (%v2) : (i32) -> ()
            db.runtime_call "DumpValue" (%marker) : (i64) -> ()
            db.runtime_call "DumpValue" (%str_const) : (!db.string) -> ()
        }
        return
	}
 }
