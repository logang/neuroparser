from PyQt4 import QtCore, QtGui, QtOpenGL
import gui_utils
from settings import Settings
import numpy as np
import os
from py_xenon_framegrabber import FrameGrabber

# The camera settings control panel
class CameraSettings(QtGui.QWidget):
    """
    A window that has the various camera-specific settings. These
    control acquisition from the micromanager frame grabbing thread.
    """
    def __init__(self, frameGrabber, cameraLabel, parent=None):
        QtGui.QWidget.__init__(self, parent)

        # Local copy of the singleton, thread-safe settings object
        self.settings = Settings()
        self.frameGrabber = frameGrabber
        self.cameraLabel = cameraLabel
        self.settingsNamespace = "camera/" + self.cameraLabel

        self.widgets = {}

        # WINDOW CONTROLS & WIDGETS
        #
        # Set up the window and the basic controls.
        self.title = gui_utils.TitleWidget(label='',
                                     title='Capture from Camera',
                                     description='Stream images from a camera using supported by micro-manager.')
        self.settingsLayout = QtGui.QVBoxLayout(self)

        # CAMERA PROPERTIES
        #
        # The code below builds up a series of settings widgets
        # appropriate to the currently loaded camera. We start with
        # the list of properties supported by this camera.
        properties = self.frameGrabber.getDevicePropertyNames(self.cameraLabel)

        # Used to parse and apply property types.  These property
        # types parallel those used by micromanager, would need to be
        # updated if the enumeration for micromanager types changes in
        # future versions of that library.
        self.prop_descriptions = ("Undefined", "String", "Integer", "Float")
        self.prop_operators = (str, str, lambda x: int(float(x)), float)

        # CAMERA INFO
        #
        # This info in returned by the micromanager driver.
        self.infoGroup = QtGui.QGroupBox('Camera Information', self)
        self.infoLayout = QtGui.QGridLayout(self.infoGroup)
        row_counter = 0

        # Populate the layout with read only properties
        remaining_properties = []
        for p in properties:
            if self.frameGrabber.isPropertyReadOnly(self.cameraLabel, p):

                # Create widgets
                label = QtGui.QLabel(p)
                value = QtGui.QLabel(self.frameGrabber.getProperty(self.cameraLabel, p))

                # Add to layout
                self.infoLayout.addWidget(label, row_counter, 0)
                self.infoLayout.addWidget(value, row_counter, 1)

                row_counter += 1
            else:
                remaining_properties.append(p)

        self.infoGroup.setLayout(self.infoLayout)
        self.settingsLayout.addWidget(self.infoGroup)

        # PRIMARY PROPERTIES
        #
        # This includes important properties like exposure, gain, and offset.
        properties = remaining_properties
        remaining_properties = []

        for p in properties:
            prop_type = self.frameGrabber.getPropertyType(self.cameraLabel, p)
            default_value = self.prop_operators[prop_type](self.frameGrabber.getProperty(self.cameraLabel, p))
            allowed_values = self.frameGrabber.getAllowedPropertyValues(self.cameraLabel, p)
            lower_limit = self.prop_operators[prop_type](self.frameGrabber.getPropertyLowerLimit(self.cameraLabel, p))
            upper_limit = self.prop_operators[prop_type](self.frameGrabber.getPropertyUpperLimit(self.cameraLabel, p))

            # Exposure
            if p.lower() == 'exposure':
                if lower_limit == 0:
                    lower_limit = 1e-3
                exposures = gui_utils.ExponentialMap(lower_limit*1e-3, upper_limit*1e-3)
                self.exposureGroup = gui_utils.SliderWidget(exposures, gui_utils.TimeDisplay(),
                                                            0.25, 'Exposure', steps=999, compact=True)
                self.settingsLayout.addWidget(self.exposureGroup)
                self.connect(self.exposureGroup, QtCore.SIGNAL('valueSlid()'), self.exposureChanged)
                self.connect(self.exposureGroup, QtCore.SIGNAL('valueEdited()'), self.exposureChanged)
                self.settings[self.settingsNamespace].refreshWithDefault('exposure_ms', default_value)
                self.exposureGroup.setValue(self.settings[self.settingsNamespace].exposure_ms / 1.0e3)
                self.exposureChanged()

            elif p.lower() == 'gain':
                gains = gui_utils.LinearMap(lower_limit, upper_limit)
                self.gainGroup = gui_utils.SliderWidget(gains, (float,lambda x:'%.3g'%x), 1.0,
                                                        'Gain', steps=999, compact=True)
                self.settingsLayout.addWidget(self.gainGroup)
                self.connect(self.gainGroup, QtCore.SIGNAL('valueSlid()'), self.gainChanged)
                self.connect(self.gainGroup, QtCore.SIGNAL('valueEdited()'), self.gainChanged)
                self.settings[self.settingsNamespace].refreshWithDefault('gain', default_value)
                self.gainGroup.setValue(self.settings[self.settingsNamespace].gain)
                self.gainChanged()

            else:
                remaining_properties.append(p)

        # ADDITIONAL PROPERTIES
        #
        # Pretty much everything else.

        # Populate the layout with widgets to control camera settings.
        self.propertiesGroup = QtGui.QGroupBox('Other Camera Properties', self)
        self.propertiesLayout = QtGui.QGridLayout(self.propertiesGroup)

        row_counter = 0
        col_counter = 0
        for p in remaining_properties:
            prop_type = self.frameGrabber.getPropertyType(self.cameraLabel, p)
            default_value = self.prop_operators[prop_type](self.frameGrabber.getProperty(self.cameraLabel, p))
            allowed_values = self.frameGrabber.getAllowedPropertyValues(self.cameraLabel, p)

            # Properties with a finite number of "allowed values" get a combo box.
            if (len(allowed_values) > 0):
                self.addComboBoxProperty(p, prop_type, allowed_values, self.propertiesLayout,
                                         row_counter, col_counter)

            # Properties with "limits" are given sliders in the gui.
            elif (self.frameGrabber.hasPropertyLimits(self.cameraLabel, p)):
                self.addSliderProperty(p, prop_type, self.propertiesLayout, row_counter, col_counter)

            # Remaining widgets get a QLineEdit text box.
            else:
                self.addLineEditProperty(p, prop_type, self.propertiesLayout, row_counter, col_counter)

            # Proceed through the grid
            if (col_counter == 1):
                col_counter = 0
                row_counter += 1
            else:
                col_counter = 1

        self.propertiesGroup.setLayout(self.propertiesLayout)
        self.settingsLayout.addWidget(self.propertiesGroup)

        # Finalize the window
        self.settingsLayout.addStretch()
        self.setLayout(self.settingsLayout)


    def addComboBoxProperty(self, pname, ptype, pvalues, layout, row, col):
        # Create widgets
        self.widgets[pname] = QtGui.QComboBox(self)
        for v in pvalues:
            self.widgets[pname].addItem(v, self.prop_operators[ptype](v))

        # Set the default value
        default_value = self.frameGrabber.getProperty(self.cameraLabel, pname)
        self.settings[self.settingsNamespace].refreshWithDefault(pname, default_value)
        default_idx = self.widgets[pname].findText(self.settings[self.settingsNamespace].value(pname))
        self.widgets[pname].setCurrentIndex(default_idx)
        self.frameGrabber.setProperty(self.cameraLabel, pname, str(self.settings[self.settingsNamespace].value(pname)))

        groupBox = QtGui.QGroupBox(pname, self)
        groupLayout = QtGui.QHBoxLayout(groupBox)

        # Add to layout
        groupLayout.addWidget(self.widgets[pname], 0)
        groupBox.setLayout(groupLayout)

        layout.addWidget(groupBox, row, col)

        # Connect up signals and slots
        self.widgets[pname].currentIndexChanged[str].connect(lambda x : self.comboEvent(pname, x))

    def addSliderProperty(self, pname, ptype, layout, row, col):
        lower_limit = self.prop_operators[ptype](self.frameGrabber.getPropertyLowerLimit(self.cameraLabel, pname))
        upper_limit = self.prop_operators[ptype](self.frameGrabber.getPropertyUpperLimit(self.cameraLabel, pname))
        default_value = self.prop_operators[ptype](self.frameGrabber.getProperty(self.cameraLabel, pname))

        # Create widgets
        label = QtGui.QLabel(pname)
        val_map = gui_utils.LinearMap(lower_limit, upper_limit)

        # Specialize handling of float vs int properties
        numerical_handler = (float,lambda x:'%.2g'%x)
        self.widgets[pname] = gui_utils.SliderWidget(val_map, numerical_handler,
                                                     0.25, pname, steps=999, compact=True)
        self.settings[self.settingsNamespace].refreshWithDefault(pname, default_value)
        # print pname, self.prop_operators[ptype](self.settings[self.settingsNamespace].value(pname))
        val = self.prop_operators[ptype](self.settings[self.settingsNamespace].value(pname))
        if (val > lower_limit and val < upper_limit):
            self.widgets[pname].setValue(val)
            self.frameGrabber.setProperty(self.cameraLabel, pname,
                                          str(self.settings[self.settingsNamespace].value(pname)))

        # Add to layout
        layout.addWidget(self.widgets[pname], row, col)


        self.connect(self.widgets[pname], QtCore.SIGNAL('valueSlid()'),
                                                        lambda : self.sliderEvent(pname))
        self.connect(self.widgets[pname], QtCore.SIGNAL('valueEdited()'), 
                                                        lambda : self.sliderEvent(pname))

    def addLineEditProperty(self, pname, ptype, layout, row, col):
        # Create widgets
        # print pname, '(read only: ', self.frameGrabber.isPropertyReadOnly(self.cameraLabel, pname), ')'

        default_value = self.frameGrabber.getProperty(self.cameraLabel, pname)

        # This if clause prevents accidental resizing of the Andor
        # driver, which currently does not mark CCD sizes as read only
        # properties!
        if not self.cameraLabel == 'demo_cam' and (pname == 'OnCameraCCDXSize' or pname == 'OnCameraCCDYSize'):
            self.settings[self.settingsNamespace].setValue(pname, default_value)            
        else:
            self.settings[self.settingsNamespace].refreshWithDefault(pname, default_value)
        self.widgets[pname] = QtGui.QLineEdit(self.settings[self.settingsNamespace].value(pname), self)
        self.frameGrabber.setProperty(self.cameraLabel, pname, str(self.settings[self.settingsNamespace].value(pname)))

        groupBox = QtGui.QGroupBox(pname, self)
        groupLayout = QtGui.QHBoxLayout(groupBox)

        # Add to layout
        groupLayout.addWidget(self.widgets[pname], 0)
        groupBox.setLayout(groupLayout)

        layout.addWidget(groupBox, row, col)
        self.widgets[pname].editingFinished.connect(lambda : self.lineEditEvent(pname))

    # ----------------------- EVENT HANDLERS ----------------------------

    def comboEvent(self, pname, val):
        #print 'Combo box callback: ', pname, val
        self.frameGrabber.setProperty(self.cameraLabel, pname, str(val))
        self.settings[self.settingsNamespace].setValue(pname, str(val))

    def sliderEvent(self, pname):
        lower_limit = float(self.frameGrabber.getPropertyLowerLimit(self.cameraLabel, pname))
        upper_limit = float(self.frameGrabber.getPropertyUpperLimit(self.cameraLabel, pname))
        val = float(self.widgets[pname].value())
        val = min(max(lower_limit, val), upper_limit)
        #print 'Slider callback: ',pname,val
        self.frameGrabber.setProperty(self.cameraLabel, pname, str(val))
        self.settings[self.settingsNamespace].setValue(pname, val)

    def lineEditEvent(self, pname):
        val = self.widgets[pname].text()
        #print 'LineEdit callback: ',pname,val
        self.frameGrabber.setProperty(self.cameraLabel, pname, str(val))
        self.settings[self.settingsNamespace].setValue(pname, str(val))
        
    def exposureChanged(self):
        """
        When the exposure slider is adjusted
        """
        lower_limit = float(self.frameGrabber.getPropertyLowerLimit(self.cameraLabel, 'Exposure'))
        upper_limit = float(self.frameGrabber.getPropertyUpperLimit(self.cameraLabel, 'Exposure'))
        exposure_ms = float(self.exposureGroup.value()) * 1e3
        self.frameGrabber.setProperty(self.cameraLabel, "Exposure", str(exposure_ms))
        self.settings[self.settingsNamespace].exposure_ms = exposure_ms

    def gainChanged(self):
        """
        When the gain slider is adjusted
        """
        lower_limit = float(self.frameGrabber.getPropertyLowerLimit(self.cameraLabel, 'Gain'))
        upper_limit = float(self.frameGrabber.getPropertyUpperLimit(self.cameraLabel, 'Gain'))
        gain = float(self.gainGroup.value())
        gain = min(max(lower_limit, gain), upper_limit)
        self.frameGrabber.setProperty(self.cameraLabel, "Gain", str(gain))
        self.settings[self.settingsNamespace].gain = gain
