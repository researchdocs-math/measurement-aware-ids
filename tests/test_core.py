import pathlib,sys,unittest,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from export import meter
from prepare import canonical,label_at
from evaluate import threshold,metrics
class CoreTests(unittest.TestCase):
 def call(self,t,u,p=1,idle=60,active=600):
  n=len(t)
  return meter(np.array(t,float),np.arange(1,n+1,dtype=np.float32)*100,np.arange(n,dtype=np.int8)%2,np.full(n,2,np.uint8),np.full(n,6,np.uint8),np.array([0],np.int64),np.array([n],np.int64),np.array([0],np.int64),np.array(u,float),p,idle,active)
 def test_sampling_precedes_timeout(self):
  x,ids,t=self.call([0,4,8],[.1,.9,.1],p=.5,idle=5)
  np.testing.assert_array_equal(x[:,0],[1,1]);np.testing.assert_array_equal(x[:,1],[100,300])
 def test_active_boundary(self):
  x,_,_=self.call([0,2,4],[0,0,0],active=4)
  np.testing.assert_array_equal(x[:,0],[2,1]);self.assertEqual(x[0,2],2)
 def test_unobserved_unit(self):
  x,_,_=self.call([0,1],[.9,.9],p=.1);self.assertEqual(len(x),0)
 def test_conservation(self):
  rng=np.random.default_rng(17)
  for trial in range(30):
   n=50;uniform=rng.random(n);x,_,_=self.call(np.cumsum(rng.random(n)*10),uniform,p=.3,idle=4,active=20)
   self.assertEqual(x[:,0].sum(),(uniform<.3).sum());self.assertAlmostEqual(float(x[:,1].sum()),float((np.arange(1,n+1)*100)[uniform<.3].sum()))
 def test_stats(self):
  x,_,_=self.call([0,2,8],[0,0,0]);self.assertEqual(x[0,0],3);self.assertAlmostEqual(x[0,3],200);self.assertAlmostEqual(x[0,7],4);self.assertAlmostEqual(x[0,8],2);self.assertEqual(x[0,9],6)
 def test_direction_restarts(self):
  x,_,_=self.call([0,2,4],[.9,.1,.1],p=.5);self.assertEqual(x[0,10],.5)
 def test_label_interval(self):
  info=([1.,4.],[(1.,2.,0),(4.,5.,1)])
  self.assertEqual(label_at(info,1.5),0);self.assertEqual(label_at(info,3.),-1);self.assertEqual(label_at(info,4.5),1)
 def test_canonical(self):
  self.assertEqual(canonical(b'a',2,b'b',1,6),canonical(b'b',1,b'a',2,6))
 def test_missing_attacks_count(self):
  m=metrics(np.array([.9,-1,.1]),np.array([1,1,0]),.5,np.array([1,0,1],bool));self.assertEqual(m['recall'],.5);self.assertEqual(m['observed_recall'],1.)
 def test_threshold_ties(self):
  scores=np.array([.1]*90+[.5]*10);q=threshold(scores,np.zeros(100),.05);self.assertEqual(q,.5);self.assertEqual((scores>q).sum(),0)
if __name__=='__main__':unittest.main()
