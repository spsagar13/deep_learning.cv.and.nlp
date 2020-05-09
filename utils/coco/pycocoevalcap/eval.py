__author__ = 'tylin'

from utils.coco.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from utils.coco.pycocoevalcap.bleu.bleu import Bleu
from utils.coco.pycocoevalcap.meteor.meteor import Meteor
from utils.coco.pycocoevalcap.rouge.rouge import Rouge
from utils.coco.pycocoevalcap.cider.cider import Cider
from utils.coco.pycocoevalcap.spice.spice import Spice

import os
import pandas as pd
from config import Config


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
            (Meteor(), "METEOR")

        ]

        # =================================================
        # Compute scores
        # =================================================

        eval_results = []

        # df = pd.DataFrame({'metric': image_ids,'score': new_image_filess})

        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
                    # df = pd.DataFrame({'metric': [m], 'score': [sc]})
                    eval_results.append({'metric': m, 'score': float(sc)})
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
                # df = pd.DataFrame({'metric': [method], 'score': [score]})
                eval_results.append({'metric': method, 'score': float(score)})
        self.setEvalImgs()

        config_new = Config()

        if config_new.eval_people_score == 'Y':
            filename = 'val/' + config_new.cnn + '_people_val_score.txt'

            with open(filename, 'w') as filehandle:
                for listitem in eval_results:
                    filehandle.write('%s\n' % listitem)
            filehandle.close()

        else:
            filename = 'val/' + config_new.cnn + '_general_val_score.txt'
            with open(filename, 'w') as filehandle:
                for listitem in eval_results:
                    filehandle.write('%s\n' % listitem)
            filehandle.close()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
