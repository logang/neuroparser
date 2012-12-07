"""
The Lenslet Settings panel
"""
from PyQt4 import QtCore, QtGui, QtOpenGL
import os
from settings import Settings

class LensletSettings(QtGui.QWidget):
    """
    A window that has settings on how the image is multiplexed
    """
    def __init__(self, imageDisplayWindow, parent=None):
        QtGui.QWidget.__init__(self, parent)

        # Use this object to set and communicate application settings
        self.settings = Settings()
        self.displayWindow = imageDisplayWindow

        self.updating = False
        
        self.centerGroup = QtGui.QGroupBox('Center lenslet position', self)
        self.offsetXSpinBox = QtGui.QDoubleSpinBox()
        self.offsetXSpinBox.setRange(-20000.0,20000.0)
        self.offsetXSpinBox.setDecimals(3)
        self.offsetXSpinBox.setValue(256.5)
        self.offsetXSpinBox.setSingleStep(0.1)
        self.offsetXSpinBox.setSuffix(' pixel(s)')
        self.offsetYSpinBox = QtGui.QDoubleSpinBox()
        self.offsetYSpinBox.setRange(-20000.0,20000.0)
        self.offsetYSpinBox.setDecimals(3)
        self.offsetYSpinBox.setValue(256.5)
        self.offsetYSpinBox.setSingleStep(0.1)
        self.offsetYSpinBox.setSuffix(' pixel(s)')
        self.centerLayout = QtGui.QGridLayout(self.centerGroup)
        self.centerLayout.addWidget(QtGui.QLabel('x:'),0,0)
        self.centerLayout.addWidget(self.offsetXSpinBox,0,1)
        self.centerLayout.addWidget(QtGui.QLabel('y:'),0,3)
        self.centerLayout.addWidget(self.offsetYSpinBox,0,4)
        self.centerLayout.setColumnStretch(2,1)
        self.centerLayout.setSpacing(0)
        self.centerGroup.setLayout(self.centerLayout)

        self.horizGroup = QtGui.QGroupBox('One lenslet to the right', self)
        self.horizXSpinBox = QtGui.QDoubleSpinBox()
        self.horizXSpinBox.setDecimals(3)
        self.horizXSpinBox.setValue(17.0)
        self.horizXSpinBox.setMinimum(-1000.0)
        self.horizXSpinBox.setMaximum(1000.0)
        self.horizXSpinBox.setSingleStep(0.005)
        self.horizXSpinBox.setSuffix(' pixel(s)')
        self.horizYSpinBox = QtGui.QDoubleSpinBox()
        self.horizYSpinBox.setDecimals(3)
        self.horizYSpinBox.setValue(0.0)
        self.horizYSpinBox.setMinimum(-1000.0)
        self.horizYSpinBox.setMaximum(1000.0)
        self.horizYSpinBox.setSingleStep(0.005)
        self.horizYSpinBox.setSuffix(' pixel(s)')
        self.horizLayout = QtGui.QGridLayout(self.horizGroup)
        self.horizLayout.addWidget(QtGui.QLabel('dx:'),0,0)
        self.horizLayout.addWidget(self.horizXSpinBox,0,1)
        self.horizLayout.addWidget(QtGui.QLabel('dy:'),0,3)
        self.horizLayout.addWidget(self.horizYSpinBox,0,4)
        self.horizLayout.setColumnStretch(2,1)
        self.horizLayout.setSpacing(0)
        self.horizGroup.setLayout(self.horizLayout)

        self.vertGroup = QtGui.QGroupBox('One lenslet down', self)
        self.vertXSpinBox = QtGui.QDoubleSpinBox()
        self.vertXSpinBox.setDecimals(3)
        self.vertXSpinBox.setValue(0.0)
        self.vertXSpinBox.setMinimum(-1000.0)
        self.vertXSpinBox.setMaximum(1000.0)
        self.vertXSpinBox.setSingleStep(0.005)
        self.vertXSpinBox.setSuffix(' pixel(s)')
        self.vertYSpinBox = QtGui.QDoubleSpinBox()
        self.vertYSpinBox.setDecimals(3)
        self.vertYSpinBox.setValue(17.0)
        self.vertYSpinBox.setMinimum(-1000.0)
        self.vertYSpinBox.setMaximum(1000.0)
        self.vertYSpinBox.setSingleStep(0.005)
        self.vertYSpinBox.setSuffix(' pixel(s)')
        self.vertLayout = QtGui.QGridLayout(self.vertGroup)
        self.vertLayout.addWidget(QtGui.QLabel('dx:'),0,0)
        self.vertLayout.addWidget(self.vertXSpinBox,0,1)
        self.vertLayout.addWidget(QtGui.QLabel('dy:'),0,3)
        self.vertLayout.addWidget(self.vertYSpinBox,0,4)
        self.vertLayout.setSpacing(0)
        self.vertLayout.setColumnStretch(2,1)
        self.vertGroup.setLayout(self.vertLayout)

        spinboxToolTip = 'Enter a value, use the up/down arrows, or use mouse scroll.\nHold down Ctrl (Command on Mac) to scroll faster using the mouse.'
        self.offsetXSpinBox.setToolTip(spinboxToolTip)
        self.offsetYSpinBox.setToolTip(spinboxToolTip)
        self.horizXSpinBox.setToolTip(spinboxToolTip)
        self.horizYSpinBox.setToolTip(spinboxToolTip)
        self.vertXSpinBox.setToolTip(spinboxToolTip)
        self.vertYSpinBox.setToolTip(spinboxToolTip)

        self.buttonGroup = QtGui.QGroupBox('', self)
        self.loadButton = QtGui.QPushButton('&Load')
        self.loadButton.setToolTip('Load the lenslet parameters from a file')
        self.saveButton = QtGui.QPushButton('&Save')
        self.saveButton.setToolTip('Save the lenslet parameters to a file')
        self.recenterButton = QtGui.QPushButton('Re&center')
        self.recenterButton.setToolTip('Reset center lenslet position to default settings')
        self.resetButton = QtGui.QPushButton('&Reset')
        self.resetButton.setToolTip('Reset center lenslet position and lenslet spacing to default settings')
        self.flipVButton = QtGui.QPushButton('&Flip vertically')
        self.flipVButton.setToolTip('Change the lenslet parameters so that they correspond to a vertically-flipped image')
        self.rotate180Button = QtGui.QPushButton('R&otate 180')
        self.rotate180Button.setToolTip('Change the lenslet parameters so that they correspond to a 180-degree rotated image')
        self.showLensletsButton = QtGui.QPushButton('S&how rectification')
        self.showLensletsButton.setToolTip("Display the raw light field image as well as lenslet grid for rectification")

        self.shiftCenterButton = QtGui.QPushButton('Sh&ift center')
        self.shiftCenterButton.setToolTip("Shift the center lenslet position to as close to the center of the image as possible while still retaining the same grid")
        self.loadWarpButton = QtGui.QPushButton('Load warp')
        self.loadWarpButton.setToolTip('Load from ImageStack -lfrectify warp settings')
        self.saveWarpButton = QtGui.QPushButton('Save warp')
        self.saveWarpButton.setToolTip('Save to ImageStack -lfrectify warp settings')
        # no more ImageStack support
        self.loadWarpButton.setVisible(False)
        self.saveWarpButton.setVisible(False)
        
        self.buttonLayout = QtGui.QGridLayout(self.buttonGroup)
        self.buttonLayout.addWidget(self.loadButton,0,0)
        self.buttonLayout.addWidget(self.saveButton,0,1)
        self.buttonLayout.addWidget(self.loadWarpButton,1,0)
        self.buttonLayout.addWidget(self.saveWarpButton,1,1)
        self.buttonLayout.addWidget(self.flipVButton,2,0)
        self.buttonLayout.addWidget(self.rotate180Button,2,1)
        self.buttonLayout.addWidget(self.recenterButton,3,0)
        self.buttonLayout.addWidget(self.resetButton,3,1)
        self.buttonLayout.addWidget(self.shiftCenterButton,3,2)
        self.buttonLayout.addWidget(self.showLensletsButton,4,0)
        self.buttonLayout.setColumnStretch(3,1)
        self.buttonLayout.setRowStretch(5,1)
        self.buttonGroup.setLayout(self.buttonLayout)

        rectificationDirections = ["Click 'Show rectification' to display a grid overlay over the raw light field image.",
                                   "Click 'Recenter' if grid roughly matches lenslet images, or 'Reset' if grid is way off.",
                                   "Scroll the raw image by dragging the right mouse button (Control-drag on Mac also works) over the image until you can see the center lenslet 'box' (this is brighter than the other boxes and generally will contain a circle)",
                                   "Zoom in and out by using the mouse scroll wheel over the image or by using Ctrl with the +/- buttons (Command on Mac).",
                                   "Find the center of the closest lenslet image and adjust the center lenslet position until the center of the circle matches the center of the closest lenslet image.",
                                   "Adjust the 'One lenslet to the right' values so that the box immediately to the right of the center box roughly matches that of the lenslet immediately to the right.",
                                   "Scroll the image to the right edge and follow the horizontal lenslet grid pattern, making adjustments to 'One lenslet to the right' as necessary.",
                                   "Repeat for 'One lenslet down'.",
                                   "Save the lenslet rectification settings using 'Save lenslets'"]
        directionsText = '<br>'.join([str(i+1)+'. '+x for (x,i) in zip(rectificationDirections,range(len(rectificationDirections)))])

        self.directions = QtGui.QGroupBox('How to rectify')
        self.directionsBox = QtGui.QTextEdit(directionsText)
        self.directionsBox.setReadOnly(True)
        self.directionsLayout = QtGui.QGridLayout(self.directions)
        self.directionsLayout.addWidget(self.directionsBox,0,0)
        self.directions.setLayout(self.directionsLayout)

        self.settingsLayout = QtGui.QGridLayout(self)
        self.settingsLayout.addWidget(self.centerGroup,0,0)
        self.settingsLayout.addWidget(self.horizGroup,1,0)
        self.settingsLayout.addWidget(self.vertGroup,2,0)
        self.settingsLayout.addWidget(self.buttonGroup,3,0)
        self.settingsLayout.addWidget(self.directions,4,0)
        self.settingsLayout.setRowStretch(4,1)
        self.setLayout(self.settingsLayout)

        self.updateFromParent()

        self.connect(self.offsetXSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.offsetChanged)
        self.connect(self.offsetYSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.offsetChanged)

        self.connect(self.horizXSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.horizChanged)
        self.connect(self.horizYSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.horizChanged)
        self.connect(self.vertXSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.vertChanged)
        self.connect(self.vertYSpinBox,
                     QtCore.SIGNAL('valueChanged(double)'),
                     self.vertChanged)

        self.connect(self.loadButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.load)
        self.connect(self.saveButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.save)
        self.connect(self.recenterButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.recenter)
        self.connect(self.resetButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.reset)
        self.connect(self.flipVButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.flipVertical)
        self.connect(self.rotate180Button,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.rotate180)
        self.connect(self.showLensletsButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.showLenslets)

        self.connect(self.shiftCenterButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.shiftCenter)
        self.connect(self.loadWarpButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.loadWarp)
        self.connect(self.saveWarpButton,
                     QtCore.SIGNAL('clicked(bool)'),
                     self.saveWarp)

    def updateFromParent(self):
        self.updating = True
        # update offset from parent
        self.offsetXSpinBox.setValue(self.settings['display'].lenslet_offset[0])
        self.offsetYSpinBox.setValue(self.settings['display'].lenslet_offset[1])
        # update the horizontal and vertical basis vectors
        self.horizXSpinBox.setValue(self.settings['display'].lenslet_horiz[0])
        self.horizYSpinBox.setValue(self.settings['display'].lenslet_horiz[1])
        self.vertXSpinBox.setValue(self.settings['display'].lenslet_vert[0])
        self.vertYSpinBox.setValue(self.settings['display'].lenslet_vert[1])
        self.updating = False

    def showLenslets(self):
        pass  #

    def getLensletParams(self):
        "Get all the lenslet parameters"
        return [self.offsetXSpinBox.value(),
                self.offsetYSpinBox.value(),
                self.horizXSpinBox.value(),
                self.horizYSpinBox.value(),
                self.vertXSpinBox.value(),
                self.vertYSpinBox.value()]

    def setLensletParams(self, params):
        "Set the lenslet parameters all at once"
        self.offsetXSpinBox.setValue(params[0])
        self.offsetYSpinBox.setValue(params[1])
        self.horizXSpinBox.setValue(params[2])
        self.horizYSpinBox.setValue(params[3])
        self.vertXSpinBox.setValue(params[4])
        self.vertYSpinBox.setValue(params[5])

    def warpToLenslet(self, (x, y, dx, dy, sx, sy)):
        "Convert warp parameters to lenslet parameters"
        invDet = 1.0 / ( 1 + dx*dy)
        rightX = sx * invDet
        downX = sy*dx * invDet
        rightY = sx*dy * invDet
        downY = sy * invDet
        offsetX = (x + dx*y) * invDet + 0.5*(downX+rightX) - 0.5
        offsetY = (y + dy*x) * invDet + 0.5*(downY+rightY) - 0.5
        return (offsetX, offsetY, rightX, rightY, downX, downY)

    def lensletToWarp(self, (offsetX, offsetY, rightX, rightY, downX, downY)):
        "Convert lenslet parameters to warp parameters"
        dy = rightY/rightX
        dx = downX/downY
        det = (1+dx*dy)
        sx = rightX * det
        sy = downY * det
        tempx = (offsetX-0.5*(downX+rightX) + 0.5)*det
        tempy = (offsetY-0.5*(downY+rightY) + 0.5)*det
        invDet = 1.0/(1-dx*dy)
        x = (tempx - dx*tempy) * invDet
        y = (-dy*tempx + tempy) * invDet
        return (x, y, dx, dy, sx, sy)

    def shiftCenter(self):
        "Shift the center to as close to the center of the image as possible while still retaining the same grid"
        params = self.getLensletParams()
        width, height = self.displayWindow.textureSize
        # get center of image
        centerX = (width-1)*0.5
        centerY = (height-1)*0.5
        offsetX = centerX - params[0]
        offsetY = centerY - params[1]
        # apply inverse rotation/skew matrix
        invDet = 1.0/(params[2]*params[5] - params[3]*params[4])
        lensletX = invDet*(offsetX*params[5] - offsetY*params[4])
        lensletY = invDet*(offsetY*params[2] - offsetX*params[3])
        # round to nearest lenslet
        lensletX = round(lensletX)
        lensletY = round(lensletY)
        # apply forward matrix
        offsetX = params[2]*lensletX + params[4]*lensletY
        offsetY = params[3]*lensletX + params[5]*lensletY
        # change settings
        params[0] = offsetX + params[0]
        params[1] = offsetY + params[1]
        self.setLensletParams(params)

    def flipVertical(self):
        "Change the lenslet settings such that the original settings corresponded to a vertically flipped version of the image"
        params = self.getLensletParams()
        width, height = self.displayWindow.textureSize
        # set the centering value
        params[1] = height-1-params[1]
        # flip the sign for basis vectors
        params[3] = -params[3]
        params[4] = -params[4]
        self.setLensletParams(params)

    def rotate180(self):
        "Change the lenslet settings such that the original settings corresponded to a 180 degree rotated version of the image"
        params = self.getLensletParams()
        width, height = self.displayWindow.textureSize
        # set the centering value
        params[0] = width-1-params[0]
        params[1] = height-1-params[1]
        self.setLensletParams(params)

    def load(self):
        "Load values from a file"
        lenslet_path = self.settings['input/file'].valueWithDefault('default_path','')
        filepath = QtGui.QFileDialog.getOpenFileName(self,
                                                     'Choose a text file with lenslet settings',
                                                     lenslet_path,
                                                     'Lenslet parameter files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.settings['input/file'].valueWithDefault('default_path',new_path)
            try:
                f=open(filepath,'r')
                line=f.readline()
                f.close()
                line = line.strip()
                values2 = eval(line)
                values = [values2[i] for i in range(6)]
                self.offsetXSpinBox.setValue(values[0])
                self.offsetYSpinBox.setValue(values[1])
                self.horizXSpinBox.setValue(values[2])
                self.horizYSpinBox.setValue(values[3])
                self.vertXSpinBox.setValue(values[4])
                self.vertYSpinBox.setValue(values[5])
            except Exception, e:
                raise Error('Unable to parse lenslet settings file')

    def loadWarp(self):
        "Load values from a ImageStack lfrectify warp file"
        lenslet_path = self.settings['input/file'].valueWithDefault('default_path','')
        filepath = QtGui.QFileDialog.getOpenFileName(self,
                                                     'Choose a text file with ImageStack -lfrectify warp parameters',
                                                     lenslet_path,
                                                     'ImageStack lfrectify parameter files (*.warp);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.settings['input/file'].valueWithDefault('default_path',new_path)
            try:
                f=open(filepath,'r')
                lineTokens = [x.split() for x in f.readlines()]
                # make sure we have a correct list of parameters
                paramDict = {}
                for i in range(6):
                    assert(len(lineTokens[i]) == 3)
                    paramDict[lineTokens[i][0]+'_'+lineTokens[i][1]] = float(lineTokens[i][2])
                warpParams = (paramDict['translate_x'],
                              paramDict['translate_y'],
                              paramDict['shear_x'],
                              paramDict['shear_y'],
                              paramDict['scale_x'],
                              paramDict['scale_y'])
                params = self.warpToLenslet(warpParams)
                self.setLensletParams(params)
            except Exception, e:
                raise Error('Unable to parse ImageStack lfrectify settings file')

    def save(self):
        "Save values to a file"
        lenslet_path = self.settings['input/file'].default_path
        filepath = QtGui.QFileDialog.getSaveFileName(self,
                                                     'Please choose a text file where lenslet settings will be saved',
                                                     lenslet_path,
                                                     'Lenslet parameter files (*.txt);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.settings['input/file'].valueWithDefault('default_path',new_path)
            f=open(filepath,'w')
            f.write('(%f,%f,%f,%f,%f,%f)\n' % (self.offsetXSpinBox.value(),
                                               self.offsetYSpinBox.value(),
                                               self.horizXSpinBox.value(),
                                               self.horizYSpinBox.value(),
                                               self.vertXSpinBox.value(),
                                               self.vertYSpinBox.value()))
            f.write('# (x-offset,y-offset,right-dx,right-dy,down-dx,down-dy)')
            f.close()
            QtGui.QMessageBox.information(self,
                                          'Lenslet settings saved',
                                          'Lenslet settings have been saved to %s' % filepath)
                                        
    def saveWarp(self):
        "Save ImageStack lfrectify warp parameters to a file"
        lenslet_path = self.settings['input/file'].default_path
        filepath = QtGui.QFileDialog.getSaveFileName(self,
                                                     'Please choose a file where ImageStack -lfrectify parameters will be saved',
                                                     lenslet_path,
                                                     'ImageStack lfrectify parameter files (*.warp);;All files (*.*)')
        if filepath:
            filepath = str(filepath)
            new_path = os.path.split(filepath)[0]
            self.settings['input/file'].valueWithDefault('default_path',new_path)
            f=open(filepath,'w')
            # get the current parameters
            params = self.getLensletParams()
            # convert to warp parameters
            warpParams = self.lensletToWarp(params)
            # write it out
            f.write('translate x %lf\ntranslate y %lf\nshear x %lf\nshear y %lf \nscale x %lf\nscale y %lf\ndesired lenslet size 20\n' % warpParams)
            f.close()
            QtGui.QMessageBox.information(self,
                                          'ImageStack lfrectify parameters saved',
                                          'ImageStack lfrectify parameters have been saved to %s' % filepath)

    def reset(self):
        "Reset to default values"
        textureSize = self.displayWindow.textureSize
        self.settings['display'].lenslet_offset = ((textureSize[0]-1)*0.5, (textureSize[1]-1)*0.5)
        self.settings['display'].lenslet_horiz = (17.0,0.0)
        self.settings['display'].lenslet_vert = (0.0,17.0)
        self.emit(QtCore.SIGNAL('settingsChanged()'))
        self.updateFromParent()
        
    def recenter(self):
        "Move the grid back to the center"
        textureSize = self.displayWindow.textureSize
        self.settings['display'].lenslet_offset = ((textureSize[0]-1)*0.5, (textureSize[1]-1)*0.5)
        self.emit(QtCore.SIGNAL('settingsChanged()'))
        self.updateFromParent()
        
    def offsetChanged(self):
        if self.updating:
            return
        self.settings['display'].lenslet_offset = (self.offsetXSpinBox.value(), self.offsetYSpinBox.value())
        self.emit(QtCore.SIGNAL('settingsChanged()'))
        self.updateFromParent()
        
    def horizChanged(self):
        if self.updating:
            return
        self.settings['display'].lenslet_horiz = (self.horizXSpinBox.value(), self.horizYSpinBox.value())
        self.emit(QtCore.SIGNAL('settingsChanged()'))
        self.updateFromParent()
        
    def vertChanged(self):
        if self.updating:
            return
        self.settings['display'].lenslet_vert = (self.vertXSpinBox.value(), self.vertYSpinBox.value())
        self.emit(QtCore.SIGNAL('settingsChanged()'))
        self.updateFromParent()
