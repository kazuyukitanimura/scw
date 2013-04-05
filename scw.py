# original code https://github.com/kisa12012/classifier

import math, operator, sys

MIN_FLOAT = sys.float_info.min

NON_CATEGORY = None
NON_CATEGORY_SCORE = MIN_FLOAT

def innerProduct(featureVector, weightVector):
  score = 0.0;
  for pos, val in featureVector.iteritems():
    if len(weightVector) <= pos:
      continue
    score += weightVector[pos] * val
  return score;

def calcLossScore(scores, correct, nonCorrectPredict, margin = 0.0):
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
  return loss_score

class Vector(list):
  def resize(self, newSize, default):
    self.extend([default for _ in range(newSize - len(self))])

class Datum(object):
  def __init__(self, category, featureVector):
    self.category = category
    self.featureVector = featureVector

class SCW(object):
  def __init__(self, phi, mode=2, C=1.0):
    self.phi = phi
    self.phi2 = phi ** 2
    self.phi4 = phi ** 4
    self.mode = mode
    self.C = C
    self.covarianceMatrix = {} # key: category, value covarianceVector
    self.weightMatrix = {} # key: category, value weightVector

  def train(self, data, maxIteration):
    for _ in range(maxIteration):
      for datum in data:
        scores = self.calcScores(datum.featureVector);
        self.update(datum, scores);

  def test(self, featureVector):
    scores = self.calcScores(featureVector);
    sortedScores = sorted(scores.iteritems(), key=operator.itemgetter(1))
    return sortedScores[len(sortedScores)-1][0];

  def calcScores(self, featureVector):
    scores = {}
    scores[NON_CATEGORY_SCORE] = NON_CATEGORY
    for category, weightVector in self.weightVector.iteritems():
      scores[category] = innerProduct(featureVector, weightVector)
    return scores

  def calcV(self, datum, nonCorrectPredict):
    v = 0.0
    correctCov = self.covarianceMatrix[datum.category]
    for pos, val in datum.featureVector.iteritems():
      if len(correctCov) <= pos:
        correctCov.resize(pos + 1, 1.0)
      v += correctCov[pos] * val ** 2

    if nonCorrectPredict is NON_CATEGORY:
      return v

    wrongCov = self.covarianceMatrix[nonCorrectPredict]
    for pos, val in datum.featureVector.iteritems():
      if len(wrongCov) <= pos:
        wrongCov.resize(pos + 1, 1.0)
      v += wrongCov[pos] * val ** 2
    return v

  def calcAlpha(self, m, v):
    if self.mode == 1:
      return self.CalcAlpha1(m, v)
    elif self.mode == 2:
      return self.CalcAlpha2(m, v)
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
    u = (-alpha * v * self.phi + math.sqrt(alpha ** 2 * v ** 2 * self.phi2 + 4.0 * v)) ** 2 / 4.0
    return alpha * self.phi / (math.sqrt(u) + v * alpha * self.phi) # beta

  def update(self, datum, scores):
    nonCorrectPredict = ''
    m = - calcLossScore(scores, datum.category, nonCorrectPredict)
    v = self.calcV(datum, nonCorrectPredict)
    alpha = self.calcAlpha(m, v);
    beta = self.calcBeta(v, alpha);

    if alpha > 0.0:
      correctWeight = self.weightMatrix[datum.category]
      correctCov = self.covarianceMatrix[datum.category]
      for pos, val in datum.featureVector.iteritems():
        if len(correctWeight) <= pos:
          correctWeight.resize(pos + 1, 0.0);
        correctWeight[pos] += alpha * correctCov[pos] * val
        correctCov[pos] -= beta * val ** 2 * correctCov[pos] ** 2

      if nonCorrectPredict == NON_CATEGORY:
        return

      wrongWeight = self.weightMatrix[nonCorrectPredict]
      wrongCov = self.covarianceMatrix[nonCorrectPredict]
      for pos, val in datum.featureVector.iteritems():
        if len(wrongWeight) <= pos:
          wrongWeight.resize(pos + 1, 0.0)
        wrongWeight[pos] -= alpha * wrongCov[pos] * val
        wrongCov[pos] += beta * val ** 2 * correctCov[pos] ** 2

def main():
  scw = SCW()

if __name__=="__main__":
  main()
