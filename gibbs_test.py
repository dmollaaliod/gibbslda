"Test of the file gibbs.py"

import unittest
import gibbs

class TestGibbs(unittest.TestCase):
    doc_topics = \
        [[('bank', 1), ('money', 0), ('money', 1), ('bank', 1), ('loan', 0)], 
         [('stream', 1), ('loan', 0), ('river', 1), ('loan', 0), ('money', 1)],
         [('money', 1), ('river', 0), ('bank', 0), ('stream', 1), ('bank', 0)],
         [('bank', 1), ('stream', 1), ('bank', 0), ('bank', 1), ('river', 1)]]

    CWT = {'bank':[3,4],'money':[1,3],'loan':[3,0],'stream':[0,3],
           'river':[1,2]}

    CDT = [[2,3],[2,3],[3,2],[1,4]]

    def test_cwt(self):
        self.assertEqual(gibbs.compute_cwt('bank',1,self.doc_topics),4)
        self.assertEqual(gibbs.compute_cwt('bank',0,self.doc_topics),3)

    def test_cdt(self):
        self.assertEqual(gibbs.compute_cdt(1,0,self.doc_topics),2)
        self.assertEqual(gibbs.compute_cdt(1,1,self.doc_topics),3)

    def test_compute_CDT_CWT(self):
        self.assertEqual(gibbs.compute_CDT_CWT(self.doc_topics),
                         (self.CDT,self.CWT))

    def test_sumcwt(self):
        self.assertEqual(gibbs.compute_sumcwt(0,self.doc_topics,self.CWT),8)
        self.assertEqual(gibbs.compute_sumcwt(1,self.doc_topics,self.CWT),12)

if __name__ == "__main__":
    unittest.main()
