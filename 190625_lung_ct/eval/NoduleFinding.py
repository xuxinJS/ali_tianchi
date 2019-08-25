class NoduleFinding(object):
  '''
  Represents a nodule
  '''
  
  def __init__(self, noduleid=None, coordX=None, coordY=None, coordZ=None, coordType="World",
               CADprobability=None, noduleType=None, diameterX=None, diameterY=None, diameterZ=None,
               state=None, seriesInstanceUID=None):

    # set the variables and convert them to the correct type
    self.id = noduleid
    self.coordX = coordX
    self.coordY = coordY
    self.coordZ = coordZ
    self.coordType = coordType
    self.CADprobability = CADprobability
    # TODO
    self.noduleType = noduleType
    # self.diameter_mm = diameter
    self.diameter_x = diameterX
    self.diameter_y = diameterY
    self.diameter_z = diameterZ
    self.state = state
    self.candidateID = None
    self.seriesuid = seriesInstanceUID
