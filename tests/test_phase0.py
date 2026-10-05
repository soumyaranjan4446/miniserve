from bench.harness import percentile,summarize

def test_percentile(): assert percentile([1,2,3,4],50)==2.5
def test_summary():
    s=summarize([1,2,3]); assert s["mean"]==2; assert s["p95"]>2
