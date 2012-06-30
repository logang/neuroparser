"""
Main window for LFDisplay
"""

import os.path
from PyQt4 import QtCore, QtGui
import gui_utils
from gui.progressbar import LedProgressBar

from settings import Settings

# ----------------------------------------------------------------------------------
#                           SETTINGS PANEL MANAGEMENT
# ----------------------------------------------------------------------------------

class SettingsPanel(QtGui.QDockWidget):
    """
    A basic settings panel widget
    """
    def __init__(self, name='', message='', widget=None, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.setObjectName(name)
        self.setWindowTitle(name)
        # the default label
        self.label = QtGui.QLabel(message)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        # the stack holding the label and setting page
        self.stack = QtGui.QStackedWidget()
        self.stack.addWidget(self.label)
        # the scroller holding the stack
        self.scroller = QtGui.QScrollArea()
        self.scroller.setWidget(self.stack)
        self.scroller.setWidgetResizable(True)
        # add the scoller
        self.setWidget(self.scroller)

        if widget:
            self.placeWidget(widget)

    def placeWidget(self, widget):
        "Place a widget into this setting panel and make it active"
        index = self.stack.addWidget(widget)
        self.stack.setCurrentIndex(index)

    def removeWidget(self, widget=None):
        "Remove a widget from the setting panel"
        if not widget: widget = self.stack.currentWidget()
        if widget == self.label: return
        self.stack.removeWidget(widget)

    def widget(self):
        return self.stack.currentWidget()

class SettingsPanelManager:
    """
    A manager for all the settings panels
    """
    def __init__(self, parent):
        self._parent = parent
        self._settings = []

    def add(self, panel):
        "Add a settings panel, initially all on the right in a tab"
        if panel not in self._settings:
            if self._settings:
                self._parent.tabifyDockWidget(self._settings[-1],panel)
            else:
                self._parent.addDockWidget(QtCore.Qt.RightDockWidgetArea,
                                           panel)
            self._settings.append(panel)
        else:
            raise Error('Attempting to add the same panel twice')

    def remove(self, panel):
        if panel in self._settings:
            self._parent.removeDockWidget(panel)
            self._settings.remove(panel)
        else:
            raise Error('Attempting to remove a panel that was not added')

    def __getitem__(self, key):
        for panel in self._settings:
            if panel.windowTitle() == key:
                return panel
        return None

    def toggleViewActions(self):
        "Get a list of view actions for all the settings panels"
        actions = [x.toggleViewAction() for x in self._settings]
        for x,y in zip(actions, self._settings):
            x.setText(y.windowTitle())
        return actions

# ----------------------------------------------------------------------------------
#                               MAIN WINDOW CLASS
# ----------------------------------------------------------------------------------

class MainWindow(QtGui.QMainWindow):
    def __init__(self,  parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        # Create an application settings object (this automatically
        # shares settings with other instances of the settings object.
        self.settings = Settings()

        # Start a timer to update the progresss bars and other status items in the mainwindow at 10Hz.
        self.timerId = self.startTimer(0.1)

        # set display stuff
        self.zoom = 1.0

        # set our title
        self.setWindowIcon(QtGui.QIcon())
        self.setWindowTitle('Neuroparser')

        # create the display widget
        self.displayTabWidget = QtGui.QTabWidget(self)
#        self.imageDispWidget = image_display.ImagingDisplay(self.pluginManager, self)
#        self.displayTabWidget.addTab(self.imageDispWidget, "Raw Camera View")
#        self.pinholeDispWidget = pinhole_display.PinholeDisplay(self)
#        self.displayTabWidget.addTab(self.pinholeDispWidget, "Pinhole View")
        self.setCentralWidget(self.displayTabWidget)

        self.displayTabWidget.setCurrentIndex(self.settings['main_window'].valueWithDefault('display_tab', 0))
#        self.connect(self.displayTabWidget, QtCore.SIGNAL('currentChanged(int)'), self.displayTabChanged)

#        self.connect(self.imageDispWidget, QtCore.SIGNAL('zoomChanged(float)'), self.changeZoom)
#        self.retireQueue = self.imageDispWidget # grab from the display widget when done

#        self.connect(self.pinholeDispWidget, QtCore.SIGNAL('zoomChanged(float)'), self.changeZoom)

        # -----------------------------------------------------------------------
        #                              STATUS BAR
        # -----------------------------------------------------------------------

        # set up the status bar
        self.statusBar_ = QtGui.QStatusBar(self)
        # self.streamingStatus = QtGui.QLabel()
        # self.streamingStatus.setMargin(2)
        # self.recordingStatus = QtGui.QLabel()
        # self.recordingStatus.setMargin(2)
        # self.zoomStatus = QtGui.QLabel()
        # self.zoomStatus.setMargin(2)
        # self.recordNumStatus = QtGui.QLabel()
        # self.recordNumStatus.setMargin(2)
        # self.illuminationFpsLabel = QtGui.QLabel()
        # self.illuminationFpsLabel.setMargin(10)
        # #self.cameraBufferBar = LedProgressBar(self)
        # #self.cameraBufferBar.setCurrentValue(50)
        # self.fileIoBufferBar = LedProgressBar(self)
        # self.fileIoBufferBar.setCurrentValue(99)
        # #self.cameraBufferBar = None
        # self.setStatus() # set a default status
        # self.statusBar_.addWidget(self.streamingStatus)
        # self.statusBar_.addWidget(self.recordingStatus)
        # self.statusBar_.addWidget(self.zoomStatus)
        # self.statusBar_.addWidget(self.recordNumStatus)
        # #self.statusBar_.addPermanentWidget(QtGui.QLabel('Camera Buffer:'))
        # #self.statusBar_.addPermanentWidget(self.cameraBufferBar)
        # self.statusBar_.addPermanentWidget(QtGui.QLabel('File Buffer:'))
        # self.statusBar_.addPermanentWidget(self.fileIoBufferBar)
        self.setStatusBar(self.statusBar_)

        # -----------------------------------------------------------------------
        #                              TOOL BAR
        # -----------------------------------------------------------------------
        self.controlBar = QtGui.QToolBar(self)

        # setup our actions
 #       self.streamAction = QtGui.QAction(QtGui.QIcon(self.resource('play.png')), '&Stream', self)
 #       self.streamAction.setShortcut(QtCore.Qt.ALT + QtCore.Qt.Key_S)
 #       self.streamAction.setToolTip('Start/stop streaming frames from the camera.')
 #       self.streamAction.setCheckable(True)
 #       self.streamAction.setEnabled(True)
 #       self.connect(self.streamAction, QtCore.SIGNAL('triggered(bool)'), self.playTriggered)

        # make a pause icon
 #       self.pauseAction = QtGui.QAction(QtGui.QIcon(self.resource('pause.png')), '&Pause', self)
 #       self.pauseAction.setShortcut(QtCore.Qt.ALT + QtCore.Qt.Key_P)
 #       self.pauseAction.setToolTip('Pause/resume streaming frames from the camera.')
 #       self.pauseAction.setCheckable(True)
 #       self.pauseAction.setChecked(True)
 #       self.pauseAction.setEnabled(False)
 #       self.connect(self.pauseAction, QtCore.SIGNAL('triggered(bool)'), self.pauseTriggered)

        # make record icon
 #       self.recordAction = QtGui.QAction(QtGui.QIcon(self.resource('record.png')), '&Record', self)
 #       self.recordAction.setShortcut(QtCore.Qt.ALT + QtCore.Qt.Key_R)
 #       self.recordAction.setToolTip('Record the streamed frames to disk.')
 #       self.recordAction.setCheckable(True)
 #       self.recordAction.setChecked(False)
 #       self.recordAction.setEnabled(True)
 #       self.connect(self.recordAction, QtCore.SIGNAL('triggered(bool)'), self.record)

        # Add the quit action (not added to the gui!)
        self.quitAction = QtGui.QAction('&Quit', self)
        self.quitAction.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.quitAction.setToolTip('Exit the program.')
        self.connect(self.quitAction, QtCore.SIGNAL('triggered(bool)'), self.close)

        # Refresh the output settings.  Make sure we have good defaults.
        outputPath = self.settings['output/file'].valueWithDefault('default_path',os.path.expanduser('~'))
        experimentName = self.settings['output/file'].valueWithDefault('experiment_name','1.1')
        outputExtension = self.settings['output/file'].valueWithDefault('file_type','png')
        sequenceNumber = self.settings['output/file'].valueWithDefault('sequence_number','0')
        useSequenceNumber = self.settings['output/file'].valueWithDefault('use_sequence_number','1')

        label0 = QtGui.QLabel('Acquisition Folder:')
        label0.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.outputPathSelector = gui_utils.PathSelectorWidget(browseCaption='Please choose a folder where new output image(s) will be created.',
                                                               default=outputPath)
        
        label1 = QtGui.QLabel('Experiment Name:')
        label1.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.outputFileHeader = QtGui.QLineEdit(experimentName, self.controlBar)
        self.outputFileHeader.setMaximumWidth(100)
        self.outputFileHeader.setText(experimentName)
        label2 = QtGui.QLabel('Image Type:')
        label2.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        image_types = [('TIFF','tif'),
                       ('PNG','png'),
                       ('JPEG','jpg')]
        self.outputFileExtension = gui_utils.CustomOptionSelectorWidget(caption='',
                                                                        options=image_types,
                                                                        custom=None,
                                                                        parent=self.controlBar,
                                                                        show=False)
        self.outputFileExtension.setValue(outputExtension)
        self.appendFrameNumber = QtGui.QCheckBox('Append frame number:')
        self.appendFrameNumber.setChecked(int(useSequenceNumber))
        self.outputFrameNumber = QtGui.QLineEdit('0')
        self.outputFrameNumber.setMaximumWidth(70)
        validator = QtGui.QIntValidator(self.outputFrameNumber)
        validator.setBottom(0)
        self.outputFrameNumber.setValidator(validator)
        self.outputFrameNumber.setText(sequenceNumber)
        self.resetSequenceButton = QtGui.QPushButton('Reset')

        spacer = QtGui.QLabel('  ')
        spacer.setMinimumWidth(100)

        # set up the toolbars
        self.controlBar.setObjectName('Control Bar')
#        self.controlBar.addAction(self.streamAction)
#        self.controlBar.addAction(self.pauseAction)
#        self.controlBar.addAction(self.recordAction)
        self.controlBar.addSeparator()

        self.controlBar.addWidget(label0)
        self.controlBar.addWidget(self.outputPathSelector)
        self.controlBar.addWidget(label1)
        self.controlBar.addWidget(self.outputFileHeader)
        self.controlBar.addWidget(label2)
        self.controlBar.addWidget(self.outputFileExtension)

        self.controlBar.addSeparator()
#        self.controlBar.addWidget(self.appendFrameNumber)
#        self.controlBar.addWidget(self.outputFrameNumber)
        self.controlBar.addWidget(self.resetSequenceButton)

        self.controlBar.setAllowedAreas(QtCore.Qt.BottomToolBarArea and QtCore.Qt.TopToolBarArea)
        self.controlBar.setMovable(True)
        self.addToolBar(self.controlBar)

        # toolbar view control
        self.viewControlBarAction = self.controlBar.toggleViewAction()
        self.viewControlBarAction.setText('&Controls')

        # connect the signals
        self.connect(self.outputPathSelector, QtCore.SIGNAL('editingFinished()'), self.setOutputPath)
        self.connect(self.outputFileHeader, QtCore.SIGNAL('editingFinished()'), self.setOutputPath)
        self.connect(self.outputFileHeader, QtCore.SIGNAL('textBrowsed(QString)'), self.setOutputPath)
        self.connect(self.outputFileExtension, QtCore.SIGNAL('valueChanged()'), self.setFileType)

        self.connect(self.appendFrameNumber, QtCore.SIGNAL('stateChanged(int)'), self.useSequenceNumber)
        self.connect(self.outputFrameNumber, QtCore.SIGNAL('textChanged(QString)'), self.setSequenceNumber)
        self.connect(self.resetSequenceButton, QtCore.SIGNAL('clicked()'), self.resetSequenceNumber)

        self.setOutputPath()
        #self.diskFrameStore.setFileType(outputExtension)
        #self.diskFrameStore.setSequenceNumber(int(sequenceNumber))
        #self.diskFrameStore.useSequenceNumber(int(useSequenceNumber))

        self.setOutputControlEnable(True)

        # -----------------------------------------------------------------------
        #                              SETTINGS PANELS
        # -----------------------------------------------------------------------

        # set up the settings panels
        self.settingsManager = SettingsPanelManager(self)

        # CAMERA SETTINGS
        # from panels.CameraSettings  import CameraSettings
        # self.cameraSettings = CameraSettings(self.frameGrabber, cameraLabel, self)
        # self.settingsManager.add(SettingsPanel(name = "Camera",
        #                                        message = "",
        #                                        widget = self.cameraSettings))
        # self.connect(self.cameraSettings,
        #              QtCore.SIGNAL('settingsChanged()'),
        #              self.imageDispWidget.updateDisplay)
        # self.connect(self.cameraSettings,
        #              QtCore.SIGNAL('settingsChanged()'),
        #              self.pinholeDispWidget.updateDisplay)

        # DISPLAY SETTINGS
#        from panels.DisplaySettings import DisplaySettings
#        self.displaySettings = DisplaySettings()
#        self.settingsManager.add(SettingsPanel(name = "Display", message = "",
#                                               widget = self.displaySettings))
#        self.connect(self.displaySettings,
#                     QtCore.SIGNAL('postprocessChanged()'),
#                     self.imageDispWidget.updateDisplay)
#        self.connect(self.displaySettings,
#                     QtCore.SIGNAL('postprocessChanged()'),
#                     self.pinholeDispWidget.updateDisplay)
#        self.connect(self.imageDispWidget,
#                     QtCore.SIGNAL('gainChanged()'),
#                     self.displaySettings.updateFromParent)
#        self.connect(self.pinholeDispWidget,
#                     QtCore.SIGNAL('gainChanged()'),
#                     self.displaySettings.updateFromParent)

        
        # ILLUMINATION WINDOW
#        self.illuminationWidget = illumination.IlluminationDisplay(self.pluginManager, None)
#        self.connect(self.pluginManager,
#                     QtCore.SIGNAL('pluginChanged()'),
#                     self.illuminationWidget.updatePlugin)
#        self.connect(self.illuminationWidget,
#                     QtCore.SIGNAL('illuminationFpsChanged(int)'),
#                     self.updateIlluminationFps)

        # PLUGIN SETTINGS
#        from panels.PluginSettings  import PluginSettings
#        self.pluginSettings = PluginSettings(self.pluginManager, self.illuminationWidget)
#        self.settingsManager.add(SettingsPanel(name = "Plugins",
#                                               message = "",
#                                               widget =  self.pluginSettings))

        # LIGHT ENGINE SETTINGS
#        from panels.LightEngineSettings import LightEngineSettings
#        self.lightEngineSettings = LightEngineSettings(self)
#        self.settingsManager.add(SettingsPanel(name = "Light Engine",
#                                               message = "",
#                                               widget =  self.lightEngineSettings))
        
        # The lenslet panel is disabled for now since we now rely
        # primarily on the automatic calibration routines.  We will
        # re-enable this panel eventually as a means of overriding
        # automatic calibration, most likely.
        #
        # LENSLET/OPTICS SETTINGS
        #        self.lensletSettings = LensletSettings(self.imageispWidget)
        #self.settingsManager.add(SettingsPanel(name = "Lenslet",
        #                                       message = "",
        #                                       widget = self.lensletSettings))
        #self.connect(self.lensletSettings,
        #             QtCore.SIGNAL('settingsChanged()'),
        #             self.imageDispWidget.updateDisplay)
        #self.connect(self.lensletSettings,
        #             QtCore.SIGNAL('settingsChanged()'),
        #             self.pinholeDispWidget.updateDisplay)

#        from panels.OpticsSettings  import OpticsSettings
        #self.opticsSettings = OpticsSettings(self.pluginManager, self.illuminationWidget, self.frameGrabber)
        #self.settingsManager.add(SettingsPanel(name = "Optics",
        #                                       message = "",
        #                                       widget = self.opticsSettings))
        #self.connect(self.opticsSettings,
        #             QtCore.SIGNAL('settingsChanged()'),
        #             self.imageDispWidget.updateDisplay)
        #self.connect(self.opticsSettings,
        #             QtCore.SIGNAL('settingsChanged()'),
        #             self.pinholeDispWidget.updateDisplay)

        # Some options
        self.setTabPosition(QtCore.Qt.LeftDockWidgetArea, QtGui.QTabWidget.North)
        self.setTabPosition(QtCore.Qt.RightDockWidgetArea, QtGui.QTabWidget.North)

        # -----------------------------------------------------------------------
        #                              MENU BAR
        # -----------------------------------------------------------------------
        self.menuBar_ = QtGui.QMenuBar(self)

        # Control Menu
        self.controlMenu = self.menuBar_.addMenu('&Control')
#        self.controlMenu.addAction(self.streamAction)
#        self.controlMenu.addAction(self.pauseAction)
#        self.controlMenu.addAction(self.recordAction)

        # View Menu
        self.viewMenu = self.menuBar_.addMenu('&View')
        self.viewMenu.addAction(self.viewControlBarAction)
        self.viewMenu.addSeparator()
        for action in self.settingsManager.toggleViewActions():
            self.viewMenu.addAction(action)
        self.setMenuBar(self.menuBar_)

        # set a sensible default for window size
        self.move(QtCore.QPoint(40,80))
        self.resize(QtCore.QSize(720,480))

        # load window settings
        self.resize(self.settings['main_window'].valueWithDefault('size', self.size()))
        self.move(self.settings['main_window'].valueWithDefault('position' ,self.pos()))

        # load the previous state of the docks and toolbars
        try:
            state = QtCore.QByteArray(self.settings['main_window'].state.decode('hex'))
            self.restoreState(state)
        except AttributeError:
            # ignore
            pass

    def setOutputControlEnable(self, state):
        self.outputPathSelector.setEnabled(state)
        self.outputFileHeader.setEnabled(state)
        self.outputFileExtension.setEnabled(state)
        self.appendFrameNumber.setEnabled(state)
        self.outputFrameNumber.setEnabled(state)
        self.resetSequenceButton.setEnabled(state)

    def setOutputPath(self):
        self.settings['output/file'].default_path = self.outputPathSelector.text()
        self.settings['output/file'].experiment_name = self.outputFileHeader.text()

        if len(self.outputPathSelector.text()) > 0 and len(self.outputFileHeader.text()) > 0:
            full_path = os.path.join(self.settings['output/file'].default_path,
                                     self.settings['output/file'].experiment_name)
#            self.diskFrameStore.setFilePrefix(os.path.join(full_path,
#                                                           self.settings['output/file'].experiment_name))

    def setFileType(self):
        val = self.outputFileExtension.value()
        self.settings['output/file'].file_type = val
#        self.diskFrameStore.setFileType(val)

    def useSequenceNumber(self, val):
        self.settings['output/file'].use_sequence_number = val
#        self.diskFrameStore.useSequenceNumber(int(val))

    def setSequenceNumber(self, val):
        self.settings['output/file'].sequence_number = str(val)
#        self.diskFrameStore.setSequenceNumber(int(val))

    def resetSequenceNumber(self):
        self.outputFrameNumber.setText('0')
        self.settings['output/file'].sequence_number = '0'
#        self.diskFrameStore.setSequenceNumber(0)

    def processDisplayModeChanged(self, num):
        buttons = [self.displayRawButton,
#                   self.displayPinholeButton,
                   self.displayApertureButton]
        if num >= 0 and num < len(buttons) and not buttons[num].isChecked():
            buttons[num].click()

    def resource(self, filename):
        """
        Return the actual location of a resource file
        """
        return os.path.join(self.settings['app'].resource_path, filename)

    def updateIlluminationFps(self, fps):
        self.setStatus(illuminationFps = fps)

    def incrementRecordCounter(self):
        "Increase the recorded frame number"
        self.setStatus(recordNum = self.recordNum + 1)

    def setStatus(self, streaming=None, recording=None, zoom=None, recordNum=None, illuminationFps = None):
        """
        Handle the current status of the program
        """
        self.streaming = 0
        self.recording = 0

        lastStreaming = self.streaming
        lastRecording = self.recording

        if None == streaming:
            streaming = self.streaming
        else:
            self.streaming = streaming
        if None == recording:
            recording = self.recording
        else:
            self.recording = recording
        if None == zoom:
            zoom = self.zoom
        else:
            self.zoom = zoom
        if None == recordNum:
            recordNum = self.recordNum
        else:
            self.recordNum = recordNum

        if streaming:
            self.streamingStatus.setText('STREAMING')
        else:
            self.streamingStatus.setText('STOPPED')
        if recording:
            if recording and streaming:
                self.recordingStatus.setText('RECORDING')
            else:
                self.recordingStatus.setText('READY TO RECORD')
        else:
            self.recordingStatus.setText('NOT RECORDING')
        if zoom >= 1.0:
            self.zoomStatus.setText('Zoom: %dX' % int(zoom))
        else:
            self.zoomStatus.setText('Zoom: 1/%dX' % int(1.0/zoom))
        self.recordNumStatus.setText('Frames recorded: %d' % self.recordNum)

        if lastStreaming != self.streaming or lastRecording != self.recording:
            if lastStreaming != self.streaming:
                self.streamingChangedEvent(self.streaming)
            if lastRecording != self.recording:
                self.recordingChangedEvent(self.recording)

 #       if (illuminationFps):
 #           self.illuminationFpsLabel.setText("Illum FPS: " + str(illuminationFps))
 #       else:
 #           self.illuminationFpsLabel.setText("Illum FPS: N/A")

    # ------------------------------ EVENT CALLBACKS -----------------------------

    #def timerEvent(self, event):
        #iobuffer_pct = (float(self.diskFrameStore.getBufferNumImages()) /
        #                self.frameGrabber.getBufferTotalCapacity() * 100)

        #self.cameraBufferBar.setCurrentValue(cbuffer_pct)
        #if (self.recording):
        #    self.fileIoBufferBar.setCurrentValue(iobuffer_pct)
        #    self.outputFrameNumber.setText(str(self.diskFrameStore.sequenceNumber()))
        #else:
        #    self.fileIoBufferBar.setCurrentValue(0)

    # ---------------------------------- SLOTS ------------------------------------

    def playTriggered(self, state):
        """
        Handle when the play action is triggered
        """
        if state:
            self.pauseAction.setChecked(False)
            self.pauseAction.setEnabled(True)
            self.streamAction.setEnabled(False)
            self.setStatus(streaming=state)
            #self.frameGrabber.startContinuousSequenceAcquisition(0);  # interval (in millis)
#            self.illuminationWidget.setStreaming(True)

    def pauseTriggered(self, state):
        """
        Handle when the pause action is triggered
        """
        if state:
            self.streamAction.setChecked(False)
            self.streamAction.setEnabled(True)
            self.pauseAction.setEnabled(False)
            self.setStatus(streaming=not state)
            #self.frameGrabber.stopSequenceAcquisition()
#            self.illuminationWidget.setStreaming(False)

    def displayTabChanged(self, index):
        self.settings['main_window'].display_tab = index
                 
    def record(self, recording):
        """
        Set whether we are recording frames to disk
        """
        # if self.recording and self.diskFrameStore.getBufferNumImages() > 0:
        #     result = QtGui.QMessageBox.warning(self, "Warning",
        #                                        "There are still buffered images that need to be saved to disk. " +
        #                                        "If you terminate recording now, they will be lost. " +
        #                                        "Are you sure you want to proceed?",
        #                                        "Yes", "No", None,
        #                                        1, # Default button is 'no',
        #                                        1) # Escape button is no
        #     if result != 0:  # If the user cancelled, bail without deactivating the record button.
        #         self.recordAction.setChecked(True)
        #         return
            
        self.setStatus(recording=recording)
#        self.diskFrameStore.setEnabled(recording)
#        self.outputFrameNumber.setText(str(self.diskFrameStore.sequenceNumber()))
#        self.illuminationWidget.setRecording(recording)

    def streamingChangedEvent(self, streaming):
        """
        When streaming state is changed
        """
        self.streamAction.setChecked(streaming)
        self.pauseAction.setChecked(not streaming)
        self.emit(QtCore.SIGNAL('streamingChangedEvent(bool)'),streaming)

        # Pass the streaming changed event along to the plugin
#        current_plugin_instance = self.pluginManager.current_plugin()
#        if (current_plugin_instance):
#            current_plugin_instance.streamingChangedEvent(streaming)

    def recordingChangedEvent(self, recording):
        """
        When recording state is changed
        """
        self.recordAction.setChecked(recording)
        self.emit(QtCore.SIGNAL('recordingChangedEvent(bool)'),recording)
        self.setOutputControlEnable(not recording)

        full_path = os.path.join(self.settings['output/file'].default_path,
                                 self.settings['output/file'].experiment_name)

        if recording:
            # Create the experiment directory, copy in calibration
            # data, and start a fresh log file.
            try:
                os.makedirs(full_path)
            except OSError:
                pass  # Do nothing if the directory already exists

            # Save calibration
            uScope_frame_manager.save_calibration(full_path)

            # Start a new plugin file
            plugin_logfile = os.path.join(full_path,
                                          self.settings['output/file'].experiment_name + '.log')
            xenon_log('Starting new logfile:' + plugin_logfile)

            from log import xenon_open_logfile, xenon_close_logfile, xenon_add_logrule
            from py_xenon_core import XenonLog

            xenon_close_logfile('plugin_log')
            xenon_open_logfile('plugin_log', plugin_logfile)
            xenon_add_logrule('plugin_log', XenonLog.INFO_MESSAGE, 'plugin')
            xenon_add_logrule('plugin_log', XenonLog.INFO_MESSAGE, 'imaging')
            xenon_add_logrule('plugin_log', XenonLog.INFO_MESSAGE, 'camera::ttl')

            msg = 'Recording session started: ' + full_path
            xenon_log(msg, 'imaging')
        else:
            msg = 'Recording session complete: ' + full_path
            xenon_log(msg, 'imaging')

        # Pass the streaming changed event along to the plugin
#        current_plugin_instance = self.pluginManager.current_plugin()
#        if (current_plugin_instance):
#            current_plugin_instance.recordingChangedEvent(recording)

    def changeZoom(self, newZoom):
        """
        When zoom level is changed
        """
        self.setStatus(zoom=newZoom)

    def closeEvent(self, event):
        """
        When main window is closed
        """

        # Stop recording
        self.pauseTriggered(True)

        # shut down the illumination widget and its TTL thread
#        self.illuminationWidget.close()

        # save the state
        state = self.saveState()
        self.settings['main_window'].state = str(state.data()).encode('hex')

        # save window settings
        self.settings['main_window'].position = self.pos()
        self.settings['main_window'].size = self.size()

        # close the window
        event.accept()

        # Sync settings to disk
        self.settings.sync()

    def keyPressEvent(self, event):
        """
        Handle some shortcut keys
        """
        if event.key() == QtCore.Qt.Key_Space:
            if (self.streaming):
                self.pauseTriggered(True)
            else:
                self.playTriggered(True)
        if event.key() == QtCore.Qt.Key_Tab:
            if (self.settings['main_window'].display_tab == 0):
                self.displayTabWidget.setCurrentIndex(1)
            else:
                self.displayTabWidget.setCurrentIndex(1)
#        elif event.key() == QtCore.Qt.Key_L:
#            if self.illuminationWidget.isVisible():
#                self.illuminationWidget.setVisible(False)
#            else:
#                self.illuminationWidget.setVisible(True)
#        elif event.key() == QtCore.Qt.Key_I:  # Hack: don't know why this needs to go here for now...
#            self.imageDispWidget.toggleInterpolation()
#            self.pinholeDispWidget.toggleInterpolation()
#        else:
            # Pass along all other key press events to the
            # plugins running in the illumination widget.
#            current_plugin_instance = self.pluginManager.current_plugin()
#            if (current_plugin_instance):
#                current_plugin_instance.keyPressEvent(event)
