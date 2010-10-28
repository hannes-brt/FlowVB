import unittest
from flowvb.core._latent_variables import _LatentVariables
import numpy as np
import os.path

class TestUpdateLatentResp(unittest.TestCase):
    def testFaithful(self):
        from data.old_faithful.setup_test_data.latent_variables \
             import latent_resp as lr
        data = np.genfromtxtxt(os.path.join("data", "old_faithful",
                                            "faithful.txt"), delimiter=",")
        
