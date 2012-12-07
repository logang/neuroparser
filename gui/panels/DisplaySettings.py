"""
The Display Settings panel
"""
from PyQt4 import QtCore, QtGui, QtOpenGL
import gui_utils
from settings import Settings

class DisplaySettings(QtGui.QWidget):
    """
    A window that has the various display-specific settings
    """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        # Use this object to set and communicate application settings
        self.settings = Settings()
        self.settings['display'].refreshWithDefault('draw_projector_boundary', False)
        
        self.gainOptionsGroup = QtGui.QGroupBox('Digital gain options', self)
        self.gainOptionsLayout = QtGui.QGridLayout(self.gainOptionsGroup)
        self.automaticGain = QtGui.QCheckBox('Automatic gain control')
        self.advancedOptions = QtGui.QCheckBox('Advanced options')
        self.automaticGainTarget = gui_utils.SliderWidget(gui_utils.LinearMap(0.0,1.0),
                                                    (float,lambda x:'%.3f'%x),
                                                    0.9,
                                                    'Automatic gain target intensity',
                                                    steps=999)
        self.automaticGainTarget.setToolTip('Automatic gain will attempt to make \nthe brightest pixel have this intensity')
        self.automaticGainTarget.setVisible(False)
        self.connect(self.automaticGain,
                     QtCore.SIGNAL('toggled(bool)'),
                     self.automaticGainToggled)
        self.connect(self.advancedOptions,
                     QtCore.SIGNAL('toggled(bool)'),
                     self.advancedOptionsToggled)
        self.connect(self.automaticGainTarget,
                     QtCore.SIGNAL('valueChanged()'),
                     self.gainOptionsChanged)
        self.gainOptionsLayout.addWidget(self.automaticGain, 0, 0)
        self.gainOptionsLayout.addWidget(self.advancedOptions, 0, 1)
        self.gainOptionsLayout.addWidget(self.automaticGainTarget, 1, 0, 1, 2)
        self.gainOptionsGroup.setLayout(self.gainOptionsLayout)

        minGain = self.settings['display'].gain_minimum
        maxGain = self.settings['display'].gain_maximum
        self.gainGroup = gui_utils.SliderWidget(gui_utils.ExponentialMap(minGain,maxGain),
                                                (float,lambda x:'%.3g'%x),
                                                1.0,
                                                'Digital gain',
                                                steps=999,
                                                compact=True)
        self.connect(self.gainGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.gainChanged)

        self.offsetGroup = gui_utils.SliderWidget(gui_utils.LinearMap(-1.0,1.0),
                                            (float,lambda x:'%.3f'%x),
                                            0.0,
                                            'Digital offset',
                                            steps=999,
                                            compact=True)
        self.connect(self.offsetGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.offsetChanged)
        
        self.gammaGroup = gui_utils.SliderWidget(gui_utils.ExponentialMap(0.05,20.0),
                                           (float,lambda x:'%.3f'%x),
                                           1.0,
                                           'Digital gamma',
                                           steps=999,
                                           compact=True)

        self.connect(self.gammaGroup,
                     QtCore.SIGNAL('valueChanged()'),
                     self.gammaChanged)


        self.buttonGroup = QtGui.QGroupBox('')
        self.resetPostprocess = QtGui.QPushButton('Reset gain, offset, gamma')
        self.buttonLayout = QtGui.QGridLayout(self.buttonGroup)
        self.buttonLayout.addWidget(self.resetPostprocess,0,0)
        self.buttonLayout.setColumnStretch(1,1)
        self.buttonGroup.setLayout(self.buttonLayout)

        self.connect(self.resetPostprocess,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.resetPostprocessSettings)

        self.bgsubtractOptionsGroup = QtGui.QGroupBox('Live Background Subtraction', self)
        self.bgsubtractOptionsLayout = QtGui.QGridLayout(self.bgsubtractOptionsGroup)
        self.bgsubtract = QtGui.QCheckBox('Live Background Subtraction')
        self.bgsubtractOptionsLayout.addWidget(self.bgsubtract, 0, 0)
        self.bgsubtractOptionsLayout.setColumnStretch(1,1)
        self.bgsubtractOptionsGroup.setLayout(self.bgsubtractOptionsLayout)
        self.bgsubtract.setChecked(self.settings['display'].refreshWithDefault('background_subtraction', False))
        self.connect(self.bgsubtract,
                     QtCore.SIGNAL('toggled(bool)'),
                     self.bgsubtractToggled)

        self.gridTypeGroup = QtGui.QGroupBox('Grid type', self)
        self.gridNone = QtGui.QRadioButton('None')
        self.gridProjector = QtGui.QRadioButton('Illumination Boundary')
        self.gridCenters = QtGui.QRadioButton('Lenslet Centers')
        self.gridBoundaries = QtGui.QRadioButton('Lenslet Boundaries')
        self.gridTypeLayout = QtGui.QGridLayout(self.gridTypeGroup)
        self.gridTypeLayout.setSpacing(0)
        self.gridTypeLayout.addWidget(self.gridNone,0,0)
        self.gridTypeLayout.addWidget(self.gridProjector,1,0)
        self.gridTypeLayout.addWidget(self.gridCenters,0,1)
        self.gridTypeLayout.addWidget(self.gridBoundaries,1,1)
        self.gridTypeGroup.setLayout(self.gridTypeLayout)

        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.gainOptionsGroup,1,0)
        self.settingsLayout.addWidget(self.gainGroup,2,0)
        self.settingsLayout.addWidget(self.offsetGroup,3,0)
        self.settingsLayout.addWidget(self.gammaGroup,4,0)
        self.settingsLayout.addWidget(self.buttonGroup,5,0)
        self.settingsLayout.addWidget(self.bgsubtractOptionsGroup,6,0)
        self.settingsLayout.addWidget(self.gridTypeGroup,7,0)
        self.settingsLayout.setRowStretch(8,1)
        self.setLayout(self.settingsLayout)

        self.updateFromParent()

        self.connect(self.gridNone,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.gridTypeChanged)
        self.connect(self.gridProjector,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.gridTypeChanged)
        self.connect(self.gridCenters,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.gridTypeChanged)
        self.connect(self.gridBoundaries,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.gridTypeChanged)

    def advancedOptionsToggled(self, b):
        self.automaticGainTarget.setVisible(b)

    def resetPostprocessSettings(self):
        self.settings['display'].automatic_gain = False
        self.settings['display'].gain = 1.0
        self.gainGroup.setValue(1.0)
        self.settings['display'].offset = 0.0
        self.offsetGroup.setValue(0.0)
        self.settings['display'].gamma = 1.0
        self.gammaGroup.setValue(1.0)
        self.settings['display'].automatic_gain_target = 0.9
        self.settings['display'].automatic_gain_fallback = 0.5
        self.emit(QtCore.SIGNAL('postprocessChanged()'))

    def updateFromParent(self):
        """
        Update settings display from actual display widget
        """
        # set the gain options
        self.automaticGain.setChecked(self.settings['display'].automatic_gain)
        self.automaticGainTarget.setValue(self.settings['display'].automatic_gain_target)
        # set the gain
        gain = self.settings['display'].gain
        self.gainGroup.setValue(gain)
        self.gainGroup.setEnabled(not self.settings['display'].automatic_gain)
        # set the offset
        offset = self.settings['display'].offset
        self.offsetGroup.setValue(offset)
        # set the gamma
        gamma = self.settings['display'].gamma
        self.gammaGroup.setValue(gamma)
        
        if self.settings['display'].draw_projector_boundary:
            self.gridProjector.setChecked(True)
        else:        
            # update grid type from parent
            if not self.settings['display'].draw_grid:
                self.gridNone.setChecked(True)
            elif self.settings['display'].grid_type == 'center':
                self.gridCenters.setChecked(True)
            elif self.settings['display'].grid_type == 'lenslet':
                self.gridBoundaries.setChecked(True)

    def automaticGainToggled(self, b):
        """
        The automatic gain control was toggled
        """
        self.settings['display'].automatic_gain = b
        self.emit(QtCore.SIGNAL('postprocessChanged()'))

        # disable gain slider if needed
        self.gainGroup.setEnabled(not b)

    def bgsubtractToggled(self, b):
        """
        BG subtract mode was toggled
        """
        self.settings['display'].background_subtraction = b
        self.emit(QtCore.SIGNAL('postprocessChanged()'))

    def gainChanged(self):
        """
        When the user selects a different gain
        """
        self.settings['display'].gain = self.gainGroup.value()
        self.emit(QtCore.SIGNAL('postprocessChanged()'))

    def offsetChanged(self):
        """
        When the user selects a different offset
        """
        self.settings['display'].offset = self.offsetGroup.value()
        self.emit(QtCore.SIGNAL('postprocessChanged()'))

    def gammaChanged(self):
        """
        When the user adjusts the gamma
        """
        self.settings['display'].gamma = self.gammaGroup.value()
        self.emit(QtCore.SIGNAL('postprocessChanged()'))

    def gainOptionsChanged(self):
        """
        When the user adjusts advanced options for gain
        """
        self.settings['display'].automatic_gain_target = self.automaticGainTarget.value()
        self.emit(QtCore.SIGNAL('postprocessChanged()'))

    def gridTypeChanged(self):
        if self.gridNone.isChecked():
            self.settings['display'].draw_grid = False
            self.settings['display'].draw_projector_boundary = False
        elif self.gridProjector.isChecked():
            self.settings['display'].draw_grid = False
            self.settings['display'].draw_projector_boundary = True
        elif self.gridCenters.isChecked():
            self.settings['display'].draw_grid = True
            self.settings['display'].draw_projector_boundary = False
            self.settings['display'].grid_type = 'center'
        elif self.gridBoundaries.isChecked():
            self.settings['display'].draw_grid = True
            self.settings['display'].draw_projector_boundary = False
            self.settings['display'].grid_type = 'lenslet'
        self.emit(QtCore.SIGNAL('postprocessChanged()'))
        self.updateFromParent()

