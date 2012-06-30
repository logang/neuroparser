from PyQt4 import QtCore, QtGui, QtOpenGL
import gui_utils
import aperture
from settings import Settings
import os
from calibration.driver import calibrateImaging, calibrateIllumination
import numpy as np

# this needs to be disabled for proper focused image calculation 
APWIDTH = 16
ENABLE_PERSPECTIVE=False

class OpticsSettings(QtGui.QWidget):
    """
    A simple way to manipulate the ray transfer matrix
    """
    def __init__(self, pluginManager, illuminationDisplay, frameGrabber, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.inputQueue = None
        self.pluginManager = pluginManager
        self.illuminationDisplay = illuminationDisplay
        self.frameGrabber = frameGrabber

        # Make sure that optics properties are initialized with reasonable defaults.
        self.settings = Settings()
        self.settings['optics'].refreshWithDefault('pitch',125.0)
        self.settings['optics'].refreshWithDefault('flen',2500.0)
        self.settings['optics'].refreshWithDefault('mag',40.0)
        self.settings['optics'].refreshWithDefault('abbe',True)
        self.settings['optics'].refreshWithDefault('na',0.95)
        self.settings['optics'].refreshWithDefault('medium',1.0)
        self.settings['optics'].refreshWithDefault('focus',0.0)
        self.settings['optics'].refreshWithDefault('perspective',0.0)
        self.settings['optics'].refreshWithDefault('aperture',0.5)

        self.focusDirections='\nMoving z in the positive direction (dragging/pressing right) brings the virtual focal plane closer to the objective.'

        self.focusGroup = gui_utils.TwinInfiniteSliderWidget(5.0, 0.5, 10,
                                                       label='Focus (z)',
                                                       suffix='um',
                                                       digits=4,
                                                       eps=1e-6,
                                                       directions=self.focusDirections)
        self.connect(self.focusGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.rayTransferChanged)

        # NOT DISPLAYED
        self.perspectiveGroup = gui_utils.SliderWidget(gui_utils.LinearMap(-1.0,1.0),
                                                       gui_utils.FloatDisplay(3),
                                                       0.0,
                                                       'Perspective')
        self.connect(self.perspectiveGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.rayTransferChanged)



        self.apertureTypeGroup = QtGui.QGroupBox('Aperture diameter', self)
        self.pinholeType = QtGui.QRadioButton('&Pinhole')
        self.variableType = QtGui.QRadioButton('&Custom')
        self.fullType = QtGui.QRadioButton('&Full')
        self.connect(self.pinholeType, QtCore.SIGNAL('clicked()'),
                     self.apertureChanged)
        self.connect(self.variableType, QtCore.SIGNAL('clicked()'),
                     self.apertureChanged)
        self.connect(self.fullType, QtCore.SIGNAL('clicked()'),
                     self.apertureChanged)
        self.apertureTypeLayout = QtGui.QGridLayout(self.apertureTypeGroup)
        self.apertureTypeLayout.addWidget(self.pinholeType, 0, 0)
        self.apertureTypeLayout.addWidget(self.variableType, 0, 1)
        self.apertureTypeLayout.addWidget(self.fullType, 0, 2)
        self.apertureTypeGroup.setLayout(self.apertureTypeLayout)

        self.apertureGroup = gui_utils.SliderWidget(gui_utils.LinearMap(0.0,1.0),
                                              gui_utils.PercentDisplay(1),
                                              0.5,
                                              'Custom effective aperture',
                                              suffix='%',
                                              steps=1000)
        self.connect(self.apertureGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.apertureChanged)

        self.pinholeType.click()

        self.apertureSamplesGroup = gui_utils.SliderWidget(gui_utils.LinearIntMap(1,APWIDTH*APWIDTH),
                                                           (int,str),
                                                           int(APWIDTH*APWIDTH/2),
                                                           'Number of aperture samples for non-pinhole rendering',
                                                           steps=APWIDTH*APWIDTH)

        self.normalizedAperture = aperture.getCircularAperture(self.apertureSamplesGroup.value())
        self.connect(self.apertureSamplesGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.samplesChanged)

        self.buttonGroup = QtGui.QGroupBox('', self)
        self.centerButton = QtGui.QPushButton('&Reset pan')
        self.centerButton.setToolTip('Recenter the panning')
        self.resetFocusButton = QtGui.QPushButton('Reset &focus')
        self.resetFocusButton.setToolTip('Set the focus back to 0.0')
        self.buttonLayout = QtGui.QGridLayout(self.buttonGroup)
        self.buttonLayout.addWidget(self.centerButton,0,0)
        self.buttonLayout.addWidget(self.resetFocusButton,0,1)
        self.buttonLayout.setColumnStretch(2,1)
        self.buttonLayout.setRowStretch(1,1)
        self.buttonGroup.setLayout(self.buttonLayout)

        self.recipeGroup = QtGui.QGroupBox('Optics Recipe', self)
        self.pitchSelector = gui_utils.CustomOptionSelectorWidget('Microlens Array Pitch:',
                                                            [('62.5um', 62.5),
                                                             ('125um', 125.0),
                                                             ('250um', 250.0)],
                                                            'Custom', float, self)
        self.pitchSelector.setValue(self.settings['optics'].pitch)
        self.flenSelector = gui_utils.CustomOptionSelectorWidget('Microlens Focal Length:',
                                                           [('1600um', 1600.0),
                                                            ('2500um', 2500.0),
                                                            ('3750um', 3750.0)],
                                                            'Custom', float, self)
        self.flenSelector.setValue(self.settings['optics'].flen)
        self.magSelector = gui_utils.CustomOptionSelectorWidget('Objective Magnification:',
                                                          [('10X',10.0),
                                                           ('20X',20.0),
                                                           ('40X',40.0),
                                                           ('60X',60.0),
                                                           ('63X',63.0),
                                                           ('100X',100.0)],
                                                          'Custom', float,
                                                          self)
        self.magSelector.setValue(self.settings['optics'].mag)
        self.abbeSelector = QtGui.QGridLayout()
        self.abbeCheckBox = QtGui.QCheckBox('Paraxial approximation', self)
        self.abbeCheckBox.setChecked(not self.settings['optics'].abbe)
        self.abbeSelector.addWidget(self.abbeCheckBox, 0, 1)
        self.abbeSelector.setColumnStretch(0,1)
        self.naSelector = gui_utils.CustomOptionSelectorWidget('Objective NA:',
                                                         [('0.45',0.45),
                                                          ('0.8',0.8),
                                                          ('0.95',0.95),
                                                          ('1.0',1.0),
                                                          ('1.3',1.3)],
                                                         'Custom', float,
                                                         self)
        self.naSelector.setValue(self.settings['optics'].na)
        (minLabel,maxLabel) = self.apertureGroup.rangeLabels(defaults=True)
        self.apertureGroup.setRangeLabels(None, maxLabel+'(%g NA)' % self.settings['optics'].na)
        self.mediumSelector = gui_utils.CustomOptionSelectorWidget('Medium Refractive Index:',
                                                             [('Dry (air)',1.0),
                                                              ('Water',1.333),
                                                              ('Oil',1.5),
                                                              ],
                                                             'Custom', float,
                                                             self)
        self.mediumSelector.setValue(self.settings['optics'].medium)
        self.recipeLayout = QtGui.QGridLayout(self.recipeGroup)
        num_items = 0
        self.recipeLayout.addWidget(self.pitchSelector,num_items,0)
        self.connect(self.pitchSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        num_items += 1 
        self.recipeLayout.addWidget(self.flenSelector,num_items,0)
        self.connect(self.flenSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        num_items += 1 
        self.recipeLayout.addWidget(self.magSelector,num_items,0)
        self.connect(self.magSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        num_items += 1 
        self.recipeLayout.addLayout(self.abbeSelector,num_items,0)
        self.connect(self.abbeCheckBox,
                     QtCore.SIGNAL('stateChanged(int)'),
                     self.recipeChanged)
        num_items += 1 
        self.recipeLayout.addWidget(self.naSelector,num_items,0)
        self.connect(self.naSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        self.connect(self.naSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.naChanged)
        num_items += 1 
        self.recipeLayout.addWidget(self.mediumSelector,num_items,0)
        self.connect(self.mediumSelector,
                     QtCore.SIGNAL('valueChanged()'),
                     self.recipeChanged)
        num_items += 1

        self.recipeButtons = QtGui.QGridLayout()
        self.recipeNote = QtGui.QLabel('')
        if self.settings['optics'].na >= self.settings['optics'].medium:
            self.recipeNote.setText('<font color="red">Error: NA >= medium index</font>')
        self.loadRecipeButton = QtGui.QPushButton('Load')
        self.saveRecipeButton = QtGui.QPushButton('Save')
        self.recipeButtons.addWidget(self.recipeNote, 0, 0)
        self.recipeButtons.addWidget(self.loadRecipeButton, 0, 1)
        self.recipeButtons.addWidget(self.saveRecipeButton, 0, 2)
        self.recipeButtons.setColumnStretch(0,1)

        self.recipeLayout.addLayout(self.recipeButtons,num_items,0)
        num_items += 1
        
        self.recipeLayout.setColumnStretch(1,1)
        self.recipeLayout.setRowStretch(num_items,1)
        self.recipeGroup.setLayout(self.recipeLayout)


        # ----------------------- CALIBRATION GROUP -------------------------
        self.calibrationOptionsGroup = QtGui.QGroupBox('Calibration', self)
        self.calibrationOptionsLayout = QtGui.QGridLayout(self.calibrationOptionsGroup)
        
        self.calibrateImagingButton = QtGui.QPushButton('Calibrate Imaging')
        self.calibrateImagingButton.setToolTip('Calibrate the Imaging Path')
        self.calibrateIlluminationButton = QtGui.QPushButton('Calibrate Illumination')
        self.calibrateIlluminationButton.setToolTip('Calibrate the Illumination Path')

        self.calibrationOptionsLayout.addWidget(self.calibrateImagingButton,0,0)
        self.calibrationOptionsLayout.addWidget(self.calibrateIlluminationButton,0,1)
        self.calibrationOptionsLayout.setColumnStretch(2,1)
        self.calibrationOptionsLayout.setRowStretch(1,1)
        self.calibrationOptionsGroup.setLayout(self.calibrationOptionsLayout)

        # ----------------------- OVERALL PANEL LAYOUT -------------------------
        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.focusGroup,0,0)
        self.settingsLayout.addWidget(self.perspectiveGroup,1,0)
        self.settingsLayout.addWidget(self.apertureTypeGroup,2,0)
        self.settingsLayout.addWidget(self.apertureGroup,3,0)
        self.settingsLayout.addWidget(self.apertureSamplesGroup,4,0)
        self.settingsLayout.addWidget(self.buttonGroup,5,0)
        self.settingsLayout.addWidget(self.recipeGroup,6,0)
        self.settingsLayout.addWidget(self.calibrationOptionsGroup,7,0)

        if not ENABLE_PERSPECTIVE:
            self.perspectiveGroup.setVisible(False)
        
        self.settingsLayout.setRowStretch(7,1)
        self.setLayout(self.settingsLayout)

#        self.connect(self.centerButton,            # ???????
#                     QtCore.SIGNAL('clicked()'),
#                     self.displayWindow.setUV)
        self.connect(self.resetFocusButton,
                     QtCore.SIGNAL('clicked()'),
                     self.resetFocus)

        self.connect(self.loadRecipeButton,
                     QtCore.SIGNAL('clicked()'),
                     self.loadRecipe)
        self.connect(self.saveRecipeButton,
                     QtCore.SIGNAL('clicked()'),
                     self.saveRecipe)

        self.connect(self.calibrateImagingButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.calibrateImaging)
        self.connect(self.calibrateIlluminationButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.calibrateIllumination)

    def calibrateImaging(self):
        calibrateImaging(self.frameGrabber, self.pluginManager)

    def calibrateIllumination(self):
        calibrateIllumination(self.frameGrabber, self.pluginManager, self.illuminationDisplay)

    def rayTransferChanged(self):
        """
        When the user selects a different focal plane or perspective
        """
        self.settings['optics'].focus = self.focusGroup.value()
        self.settings['optics'].perspective = self.perspectiveGroup.value()
        self.emit(QtCore.SIGNAL('settingsChanged()'))

    def processDisplayModeChanged(self, num):
        """
        Process a display mode change
        """
        # disable/enable focus slider
        if self.pinholeType.isChecked() or num != 2:
            self.focusGroup.setEnabled(False)
            self.focusGroup.setToolTip('Set aperture type to be something other than pinhole,\nand make sure display mode is in 3D (light field)\nto adjust focus in light field rendering')
        else:
            self.focusGroup.setEnabled(True)
            self.focusGroup.setToolTip('')
        self.apertureGroup.setEnabled(self.variableType.isChecked())
        if num == 1:
            # reset the focus and pan
            self.centerButton.click()
            self.resetFocusButton.click()
            # select pinhole
            self.pinholeType.setChecked(True)
            self.apertureChanged()
            
    def updateFocus(self):
        """
        Update the state of the focus widget based on the
        current conditions
        """
        if self.pinholeType.isChecked() or self.displayWindow.shaderNext != 2:
            self.focusGroup.setEnabled(False)
            self.focusGroup.setToolTip('Set aperture type to be something other than pinhole,\nand make sure display mode is in 3D (light field)\nto adjust focus in light field rendering')
        else:
            self.focusGroup.setEnabled(True)
            self.focusGroup.setToolTip('')
        self.apertureGroup.setEnabled(self.variableType.isChecked())

    def maxNormalizedSlope(self):
        """
        Return the maximum slope afforded by the optical system
        0.5 means at the edge of a lenslet image
        """
        imagena = self.settings['optics'].na / self.settings['optics'].mag
        if imagena < 1.0:
            ulenslope = 1.0 * self.settings['optics'].pitch / self.settings['optics'].flen
            naslope = imagena / (1.0-imagena*imagena)**0.5
            return naslope / ulenslope
        else:
            return 0.0

    def apertureChanged(self):
        """
        When the user selects a different aperture size
        """
        self.updateFocus()
        if self.pinholeType.isChecked():
            aperture = [(0.0,0.0,1.0)]
        else:
            # tell the view mode toolbar that we need to go to full 3D mode
            self.emit(QtCore.SIGNAL('displayModeChanged(int)'), 2)
            if self.variableType.isChecked():
                apertureDiameter = self.apertureGroup.value()
            else:
                apertureDiameter = 1.0
            apertureScale = apertureDiameter * self.maxNormalizedSlope() / 0.5
            aperture = [(x*apertureScale,y*apertureScale,w) for (x,y,w) in self.normalizedAperture]
            if not aperture:
                aperture = [(0.0,0.0,1.0)]
        self.settings['optics'].apertur = aperture
        self.emit(QtCore.SIGNAL('settingsChanged()'))

    def recipeChanged(self):
        # check for valid numbers
        errors = []
        try:
            pitch=self.pitchSelector.value()
        except ValueError:
            errors.append('pitch')
        try:
            flen=self.flenSelector.value()
        except ValueError:
            errors.append('focal length')
        try:
            mag=self.magSelector.value()
        except ValueError:
            errors.append('magnification')
        try:
            na=self.naSelector.value()
        except ValueError:
            errors.append('NA')
        try:
            medium=self.mediumSelector.value()
        except ValueError:
            errors.append('medium index')
        if errors:
            self.recipeNote.setText('<font color="red">Error: invalid '+', '.join(errors)+'</font>')
        elif na >= medium:
            self.recipeNote.setText('<font color="red">Error: NA >= medium index</font>')
        else:
            self.recipeNote.setText('')
            self.settings['optics'].pitch = pitch
            self.settings['optics'].flen = flen
            self.settings['optics'].mag = mag
            self.settings['optics'].abbe = not self.abbeCheckBox.isChecked()
            self.settings['optics'].na = na
            self.settings['optics'].medium = medium
            self.emit(QtCore.SIGNAL('settingsChanged()'))

    def samplesChanged(self):
        self.normalizedAperture = aperture.getCircularAperture(self.apertureSamplesGroup.value())
        self.apertureChanged()
        
    def resetFocus(self):
        self.focusGroup.setValue(0.0)

    def naChanged(self):
        "If the NA has been changed, update effective aperture slider"
        (minLabel,maxLabel) = self.apertureGroup.rangeLabels(defaults=True)
        self.apertureGroup.setRangeLabels(None, maxLabel+'(%g NA)' % self.naSelector.value())        

    def loadRecipe(self):
        "Load optics recipe from a file"
        lenslet_path = self.settings['input/file'].valueWithDefault('default_path','')
        filepath = QtGui.QFileDialog.getOpenFileName(self,
                                                     'Choose a configuration file for optics recipe',
                                                     lenslet_path,
                                                     'Text files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.settings['input/file'].default_path = new_path
            try:
                f=open(filepath,'r')
                lineTokens = [x.split() for x in f.readlines()]
                # make sure we have a correct list of parameters
                paramDict = {}
                for i in range(6):
                    assert(len(lineTokens[i]) == 2)
                    paramDict[lineTokens[i][0]] = lineTokens[i][1]
                self.settings['optics'].pitch = float(paramDict['pitch'])
                self.settings['optics'].flen = float(paramDict['flen'])
                self.settings['optics'].mag = float(paramDict['mag'])
                self.settings['optics'].abbe = (paramDict['abbe'].lower() in ['true','yes','y'])
                self.settings['optics'].na = float(paramDict['na'])
                self.settings['optics'].medium = float(paramDict['medium'])
                self.emit(QtCore.SIGNAL('settingsChanged()'))
            except Exception, e:
                QtGui.QMessageBox.critical(self,
                                           'Error',
                                           'Unable to parse optics recipe file')
                raise RuntimeError('Unable to parse optics recipe file')
            self.pitchSelector.setValue(self.settings['optics'].pitch)
            self.flenSelector.setValue(self.settings['optics'].flen)
            self.magSelector.setValue(self.settings['optics'].mag)
            self.abbeCheckBox.setChecked(not self.settings['optics'].abbe)
            self.naSelector.setValue(self.settings['optics'].na)
            self.mediumSelector.setValue(self.settings['optics'].medium)

    def saveRecipe(self):
        "Save optics recipe to a file"
        if (self.settings['input/file'].contains('default_path')):
            lenslet_path = self.settings['input/file'].default_path
        else:
            lenslet_path = os.path.expanduser('~')
        filepath = QtGui.QFileDialog.getSaveFileName(self,
                                                     'Please choose a file where optics recipe will be saved',
                                                     lenslet_path,
                                                     'Text files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.settings['input/file'].default_path = new_path
            f=open(filepath,'w')
            # write it out
            f.write('pitch %lg\n' % self.settings['optics'].pitch)
            f.write('flen %lg\n' % self.settings['optics'].flen)
            f.write('mag %lg\n' % self.settings['optics'].mag)
            f.write('abbe %s\n' % ['false','true'][self.settings['optics'].abbe])
            f.write('na %lg\n' % self.settings['optics'].na)
            f.write('medium %lg\n' % self.settings['optics'].medium)
            f.close()
            QtGui.QMessageBox.information(self, 'Optics recipe saved',
                                          'Optics recipe has been saved to %s' % filepath)

        
