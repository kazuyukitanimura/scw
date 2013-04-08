# original code https://github.com/kisa12012/classifier

import math, operator, sys
from functools import partial

MAX_FLOAT = sys.float_info.max

NON_CATEGORY = None
NON_CATEGORY_SCORE = -MAX_FLOAT

def innerProduct(featureVector, weightVector):
  score = 0.0;
  for pos, val in featureVector.iteritems():
    score += weightVector[pos] * val
  return score;

def calcLossScore(scores, correct, margin = 0.0):
  correctDone = False
  predictDone = False
  loss_score = margin
  for category, score in scores.iteritems():
    if category == correct:
      loss_score -= score
      correctDone = True
    elif not predictDone:
      nonCorrectPredict = category
      if nonCorrectPredict != NON_CATEGORY:
        loss_score += score
        predictDone = True
    if correctDone and predictDone:
      break
  return (-loss_score, nonCorrectPredict)

class Vector(dict):
  def __init__(self, default):
    self.DEFAULT = default
    return super(self.__class__, self).__init__()

  def __getitem__(self, key):
    if key not in self:
      self[key] = self.DEFAULT
    return super(self.__class__, self).__getitem__(key)

class Matrix(dict):
  def __init__(self, default):
    self.DEFAULT = default
    return super(self.__class__, self).__init__()
    
  def __getitem__(self, category):
    if category not in self:
      self[category] = Vector(self.DEFAULT)
    return super(self.__class__, self).__getitem__(category)

class Datum(object):
  def __init__(self, category, featureVector):
    self.category = category
    self.featureVector = featureVector

class SCW(object):
  def __init__(self, phi, C=1.0, mode=2):
    self.phi = phi
    self.phi2 = phi ** 2
    self.phi4 = phi ** 4
    self.mode = mode
    self.C = C
    self.covarianceMatrix = Matrix(1.0) # key: category, value covarianceVector
    self.weightMatrix = Matrix(0.0) # key: category, value weightVector

  def train(self, dataGen, maxIteration):
    for _ in range(maxIteration):
      for datum in dataGen():
        scores = self.calcScores(datum.featureVector);
        self.update(datum, scores);

  def test(self, featureVector):
    scores = self.calcScores(featureVector);
    maxScore = NON_CATEGORY_SCORE
    maxCategory = NON_CATEGORY
    for category, value in scores.iteritems():
      if maxScore < value:
        maxScore = value
        maxCategory = category
    return maxCategory

  def calcScores(self, featureVector):
    scores = {}
    scores[NON_CATEGORY] = NON_CATEGORY_SCORE
    for category, weightVector in self.weightMatrix.iteritems():
      scores[category] = innerProduct(featureVector, weightVector)
    return scores

  def calcV(self, datum, nonCorrectPredict):
    v = 0.0
    correctCov = self.covarianceMatrix[datum.category]
    for pos, val in datum.featureVector.iteritems():
      v += correctCov[pos] * val ** 2

    if nonCorrectPredict == NON_CATEGORY:
      return v

    wrongCov = self.covarianceMatrix[nonCorrectPredict]
    for pos, val in datum.featureVector.iteritems():
      v += wrongCov[pos] * val ** 2
    return v

  def calcAlpha(self, m, v):
    if self.mode == 1:
      return self.calcAlpha1(m, v)
    elif self.mode == 2:
      return self.calcAlpha2(m, v)
    return 0.0

  def calcAlpha1(self, m, v):
    psi = 1.0 + self.phi2 / 2.0;
    zeta = 1.0 + self.phi2;
    alpha = (-m * psi + math.sqrt(m ** 2 * self.phi4 / 4.0 + v * self.phi2 * zeta)) / (v * zeta)
    return min(max(alpha, 0.0), self.C) # assuming self.C > 0.0

  def calcAlpha2(self, m, v):
    n = v + 1.0 / (2.0 * self.C)
    gamma = self.phi * math.sqrt(self.phi2 * m ** 2 * v ** 2 + 4 * n * v * (n + v * self.phi2))
    alpha = (- (2.0 * m * n + self.phi2 * m * v) + gamma) / (2.0 * (n ** 2 + n * v * self.phi2))
    return max(alpha, 0.0)

  def calcBeta(self, v, alpha):
    u_sqrt = (-alpha * v * self.phi + math.sqrt(alpha ** 2 * v ** 2 * self.phi2 + 4.0 * v)) / 2.0
    return alpha * self.phi / (u_sqrt + v * alpha * self.phi) # beta

  def update(self, datum, scores):
    (m, nonCorrectPredict) = calcLossScore(scores, datum.category)
    v = self.calcV(datum, nonCorrectPredict)
    alpha = self.calcAlpha(m, v);
    beta = self.calcBeta(v, alpha);

    if alpha > 0.0:
      correctWeight = self.weightMatrix[datum.category]
      correctCov = self.covarianceMatrix[datum.category]
      for pos, val in datum.featureVector.iteritems():
        correctWeight[pos] += alpha * correctCov[pos] * val
        correctCov[pos] -= beta * val ** 2 * correctCov[pos] ** 2

      if nonCorrectPredict == NON_CATEGORY:
        return

      wrongWeight = self.weightMatrix[nonCorrectPredict]
      wrongCov = self.covarianceMatrix[nonCorrectPredict]
      for pos, val in datum.featureVector.iteritems():
        wrongWeight[pos] -= alpha * wrongCov[pos] * val
        wrongCov[pos] += beta * val ** 2 * correctCov[pos] ** 2

def parseFile(filePath):
  for line in open(filePath, 'r'):
    pieces = line.strip().split(' ')
    category = pieces.pop(0)
    featureVector = {}
    for kv in pieces:
      (k, v) = kv.split(':')
      featureVector[k] = float(v)
    datum = Datum(category, featureVector)
    yield datum

def main():
  trainPath = sys.argv[1]
  testPath = sys.argv[2]
  train = partial(parseFile, trainPath)
  test = partial(parseFile, testPath)

  mode = int(sys.argv[3]) if len(sys.argv) > 3 else None
  maxIteration = int(sys.argv[4]) if len(sys.argv) > 4 else 1

  eta = 10.0#100.0
  for _ in range(5):
    C = 1.0
    for _ in range(10):
      print "eta: %f" % eta
      print "C: %f" % C
      args = [eta, C]
      if isinstance(mode, int):
        args.append(mode)
      scw = SCW(*args)
      scw.train(train, maxIteration)
      success = 0
      testSize = 0
      for datum in test():
        testSize += 1
        if datum.category == scw.test(datum.featureVector):
          success += 1
      #print "accuracy: %f \n" % (100.0 * success / testSize)
      print "accuracy: %d / %d \n" % (success, testSize)
      break
      C *= 0.5
    break
    eta *= 0.1

if __name__=="__main__":
  main()
