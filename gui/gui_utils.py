"""
Some generic Qt GUI elements
"""

import os.path
from PyQt4 import QtCore, QtGui

import math
import controls
import components
from gui.scale import *

class Error(Exception):
    pass

def FloatDisplay(digits):
    """
    A basic floating point display for SliderWidget
    """
    strToValue = float
    formatString = '%%.%df' % digits
    valueToStr = lambda x: formatString % x
    return strToValue, valueToStr

def PercentDisplay(digits):
    """
    A percentage display for SliderWidget
    """
    def strToValue(s):
        return float(s)/100.0
    def valueToStr(val):
        return ('%%.%df' % digits) % (val*100.)
    return strToValue, valueToStr

def TimeDisplay():
    """
    A time display for SliderWidget
    """
    def strToValue(s):
        if s.endswith('us'):
            return float(s[0:-2]) * 1e-6
        elif s.endswith('ms'):
            return float(s[0:-2]) * 1e-3
        elif s.endswith('s'):
            return float(s[0:-1])
        elif s.endswith('m'):
            return float(s[0:-1]) * 60.0
        elif s.endswith('h'):
            return float(s[0:-1]) * 3600.0
        else:
            return float(s)
    def valueToStr(val):
        if val < 1e-3:
            return '%.4g us' % (val*1e6)
        elif val < 1:
            return '%.4g ms' % (val*1e3)
        elif val < 60:
            return '%.4g  s' % (val)
        elif val < 3600:
            return '%.4g  m' % (val/60.0)
        else:
            return '%.4g  h' % (val/3600.0)
    return strToValue, valueToStr

def _toSliderValue(minimum,maximum,fract):
    "Convert a fractional value to slider value"
    return int(round(fract*(maximum-minimum)))

def _toSliderValue2(slider,fract):
    return _toSliderValue(slider.minimum(),slider.maximum(),fract)

def _fromSliderValue(minimum,maximum,sliderValue):
    return 1.*(sliderValue-minimum)/(maximum-minimum)

def _fromSliderValue2(slider):
    return _fromSliderValue(slider.minimum(),slider.maximum(),slider.value())

def _extract(prefix, suffix, value):
    "Extract the actual value from a value with a prefix and suffix"
    value = str(value)
    if prefix and value.startswith(prefix):
        value = value[len(prefix):]
    if suffix and value.endswith(suffix):
        value = value[:-len(suffix)]
    return value

class SliderWidget(QtGui.QGroupBox):
    """
    A widget that has a customizable slider and display
    """
    def __init__(self, (fractToValue, valueToFract)=(float, float),
                 (strToValue, valueToStr)=(float, str),
                 defaultValue=None, label='Slider', prefix='', suffix='', steps=1001, compact=False, parent=None):
        """
        Instantiate a slider system

        label - the name of the slider system
        fractToValue - a function: [0.0,1.0] -> value
        valueToFract - a function: value -> [0.0,1.0]
        strToValue - a function that converts a string to a value
        valueToStr - a function that converts a value to a string
        prefix - a string to be prepended to the value string automatically
        suffix - a string to be appended to the value string automatically
        """
        QtGui.QGroupBox.__init__(self, label, parent)
        # save settings
        self.prefix = prefix
        self.suffix = suffix
        self.curValue = defaultValue

        self.fractToValue = fractToValue
        self.valueToFract = valueToFract
        self.strToValue = strToValue
        self.valueToStr = valueToStr

        self.compact = compact

        # set up different backgrounds
        self.defaultPalette = QtGui.QPalette()
        self.errorPalette = QtGui.QPalette()
        self.errorPalette.setColor(QtGui.QPalette.Base,QtCore.Qt.red)
        self.editPalette = QtGui.QPalette()

        # create the gui
        self.groupLayout = QtGui.QGridLayout()

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0,steps-1)

        self.slider.setValue(_toSliderValue2(self.slider, self.valueToFract(defaultValue)))
        self.minLabel = QtGui.QLabel(self.prefix+self.valueToStr(self.fractToValue(0.0))+self.suffix)
        self.maxLabel = QtGui.QLabel(self.prefix+self.valueToStr(self.fractToValue(1.0))+self.suffix)
        self.maxLabel.setAlignment(QtCore.Qt.AlignRight)

        self.curLabel = QtGui.QLineEdit(self.prefix+self.valueToStr(defaultValue)+self.suffix)
        self.curLabel.setPalette(self.defaultPalette)
        self.curLabel.setSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Preferred)

        if self.compact:
            self.curLabel.setMaximumSize(self.curLabel.maximumSize().width(),
                                         self.maxLabel.sizeHint().height())
            self.curLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
            self.groupLayout.setVerticalSpacing(8)
            self.groupLayout.addWidget(self.slider,0,0,1,3)
            self.groupLayout.addWidget(self.minLabel,1,0)
            self.groupLayout.addWidget(self.curLabel,1,1)
            self.groupLayout.addWidget(self.maxLabel,1,2)
        else:
            self.curLabel.setMaximumSize(max(self.maxLabel.sizeHint().width(),self.minLabel.sizeHint().width())+8,max(self.minLabel.sizeHint().height(),self.maxLabel.sizeHint().height()))
            self.curLabel.setAlignment(QtCore.Qt.AlignRight)
            self.groupLayout.setVerticalSpacing(0)
            self.groupLayout.addWidget(self.slider,0,0,1,2)
            self.groupLayout.addWidget(self.minLabel,1,0)
            self.groupLayout.addWidget(self.curLabel,0,2)
            self.groupLayout.addWidget(self.maxLabel,1,1)
            self.groupLayout.setColumnStretch(2,1)
            self.groupLayout.setColumnStretch(0,0)
            self.groupLayout.setColumnMinimumWidth(0,0)
            self.groupLayout.setColumnStretch(1,1)
        self.setLayout(self.groupLayout)

        # connect the signals
        self.connect(self.slider, QtCore.SIGNAL('actionTriggered(int)'), self.handleSliderChange)
        #self.connect(self.curLabel, QtCore.SIGNAL('textEdited(QString)'), self.handleTextChange)
        self.connect(self.curLabel, QtCore.SIGNAL('editingFinished()'), self.editingFinished)

    def editingFinished(self):
        "A handler to handle when the editing is finished on the line edit"
        oldvalue = self.curValue
        newstring = _extract(self.prefix,self.suffix,self.curLabel.text())
        try:
            newvalue = self.strToValue(newstring)
            self.curLabel.setPalette(self.defaultPalette)
        except ValueError:
            # an error in parsing
            self.curLabel.setPalette(self.errorPalette)
            return
        if oldvalue != newvalue:
            self.setValue(newvalue, fromText=False)
            self.emit(QtCore.SIGNAL('valueEdited()'))

    def handleSliderChange(self):
        "A handler to handle when the slider changes"
        oldvalue = self.curValue
        newvalue = self.fractToValue(_fromSliderValue2(self.slider)) 
        if oldvalue != newvalue:
            self.setValue(newvalue, fromSlider=True)
            self.emit(QtCore.SIGNAL('valueSlid()'))

    # def handleTextChange(self):
    #     "A handler to handle when the text input changes"
    #     oldvalue = self.curValue
    #     newstring = _extract(self.prefix,self.suffix,self.curLabel.text())
    #     try:
    #         newvalue = self.strToValue(newstring)
    #         self.curLabel.setPalette(self.editPalette)
    #     except ValueError:
    #         # an error in parsing
    #         self.curLabel.setPalette(self.errorPalette)
    #         return
    #     if oldvalue != newvalue:
    #         self.setValue(newvalue, fromText=True)
    #         self.emit(QtCore.SIGNAL('valueEdited()'))

    def setValue(self, value, fromSlider=False, fromText=False):
        "A slot to change the current value"
        if value != self.curValue:
            self.curValue = value
            if not fromSlider:
                self.slider.setValue(_toSliderValue2(self.slider, self.valueToFract(self.curValue)))
            if not fromText:
                self.curLabel.setText(self.prefix+self.valueToStr(self.curValue)+self.suffix)
                self.curLabel.setPalette(self.defaultPalette)
            self.emit(QtCore.SIGNAL('valueChanged()'))

    def value(self):
        "Return the current value"
        return self.curValue

    def rangeLabels(self, defaults=False):
        "Return the current range labels"
        if defaults:
            return (self.prefix+self.valueToStr(self.fractToValue(0.0))+self.suffix,
                    self.prefix+self.valueToStr(self.fractToValue(1.0))+self.suffix)
        else:
            return (self.minLabel.text(), self.maxLabel.text())

    def setRangeLabels(self, minLabel, maxLabel):
        "Set custom labels for extremes on the range"
        if minLabel is None:
            self.minLabel.setText(self.prefix+self.valueToStr(self.fractToValue(0.0))+self.suffix)
        else:
            self.minLabel.setText(minLabel)
        if maxLabel is None:
            self.maxLabel.setText(self.prefix+self.valueToStr(self.fractToValue(1.0))+self.suffix)
        else:
            self.maxLabel.setText(maxLabel)

class TitleWidget(QtGui.QGroupBox):
    """
    A widget that displays a title and description
    """
    def __init__(self, label='Label', title='Title', description='Insert description here.', parent=None):
        QtGui.QGroupBox.__init__(self, label, parent)

        self.title = QtGui.QLabel(title)
        self.title.setFont(QtGui.QFont('Helvetica',12,QtGui.QFont.Bold))

        self.description = QtGui.QLabel(description)

        self.groupLayout = QtGui.QVBoxLayout()
        self.groupLayout.addWidget(self.title)
        self.groupLayout.addWidget(self.description)
        self.setLayout(self.groupLayout)

class PropertiesWidget(QtGui.QGroupBox):
    "A widget that shows a list of properties"
    def __init__(self, props, title='Properties', headers=('Property','Value'), parent=None):
        QtGui.QGroupBox.__init__(self, title, parent)

        self.props = props

        self.table = QtGui.QTableWidget(1,2)
        self.table.verticalHeader().hide()
        if headers:
            self.table.setHorizontalHeaderLabels(QtCore.QStringList(list(headers)))
        else:
            self.table.horizontalHeader().hide()
        self.scroll = QtGui.QScrollArea(self)
        self.scroll.setWidget(self.table)
        self.scroll.setWidgetResizable(True)

        self.groupLayout = QtGui.QVBoxLayout()
        self.groupLayout.addWidget(self.scroll)
        self.setLayout(self.groupLayout)

        self.updateFromProps()

    def updateFromProps(self):
        props = self.props.copy()
        keys = props.keys()
        keys.sort()
        self.table.setColumnCount(2)
        self.table.setRowCount(len(keys))
        for key, i in zip(keys,range(len(keys))):
            entry = QtGui.QTableWidgetItem(str(key))
            entry.setFlags(QtCore.Qt.ItemIsEnabled)
            self.table.setItem(i, 0, entry)
            entry = QtGui.QTableWidgetItem(str(props[key]))
            entry.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            entry.setToolTip(str(props[key]))
            self.table.setItem(i, 1, entry)
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()

class LogWidget(QtGui.QGroupBox):
    """
    A widget that shows a log and also behaves as a file-like
    object
    """
    def __init__(self, label='Log', logFile=None, parent=None):
        QtGui.QGroupBox.__init__(self, label, parent)

        # an optional file object 
        self.logFile = logFile

        # create the UI
        self.textEdit = QtGui.QTextEdit()
        self.textEdit.setReadOnly(True)
        
        self.groupLayout = QtGui.QVBoxLayout()
        self.groupLayout.addWidget(self.textEdit)
        self.groupLayout.setStretchFactor(self.textEdit,1)
        self.setLayout(self.groupLayout)

    def write(self, s):
        "Append information to the log"
        self.textEdit.insertPlainText(s)
        if self.logFile:
            self.logFile.write(s)
        self.textEdit.ensureCursorVisible()

    def read(self):
        "Read all the text from the log"
        return self.textEdit.toPlainText()

    def clear(self):
        "Clear the log"
        self.textEdit.clear()

class TextAndBrowseWidget(QtGui.QWidget):
    """
    A widget that has both a text input field and a browse button
    """
    def __init__(self, browseLabel='', browseFunc=None, default='', browseIcon=None, parent=None):
        """
        Instantiate a TextAndBrowseWidget

        label is the name of the field
        browseLabel is the label on the browse button
        browseFunc is the function called when the browse
                   button is pressed and either returns
                   None for no change or returns a new string
                   It takes the string in the text field as input
        default is the default value in the text field
        browseIcon is an optional icon for the button

        This widget emits the following signals:
          textChanged(QString) - when the text changes
          textEdited(QString) - when the text has been edited in the edit box
          textBrowsed(QString) - when the text has been changed by the browser

        This widget accepts the following slots:
          setText(str) - set the text in the input field
        """
        QtGui.QWidget.__init__(self, parent)

        # create the UI
        self.lineEdit = QtGui.QLineEdit(default)
        if browseIcon:
            self.browseButton = QtGui.QPushButton(browseIcon,browseLabel)
        else:
            self.browseButton = QtGui.QPushButton(browseLabel)
        self._browseFunc = browseFunc
        self.groupLayout = QtGui.QHBoxLayout()
        self.groupLayout.addWidget(self.lineEdit)
        self.groupLayout.addWidget(self.browseButton)
        self.groupLayout.setStretchFactor(self.lineEdit,1)
        self.setLayout(self.groupLayout)

        # connect the signals
        self.connect(self.lineEdit,QtCore.SIGNAL('textChanged(QString)'),
                     self._textChanged)
        self.connect(self.lineEdit,QtCore.SIGNAL('textEdited(QString)'),
                     self._textEdited)
        self.connect(self.lineEdit,QtCore.SIGNAL('editingFinished()'),
                     self._editingFinished)
        self.connect(self.browseButton,QtCore.SIGNAL('clicked(bool)'),
                     self._browseClicked)

    def _textChanged(self, s):
        self.emit(QtCore.SIGNAL('textChanged(QString)'), s)

    def _textEdited(self, s):
        self.emit(QtCore.SIGNAL('textEdited(QString)'), s)

    def _editingFinished(self):
        self.emit(QtCore.SIGNAL('editingFinished()'))

    def _textBrowsed(self, s):
        self.lineEdit.setText(s)
        self.emit(QtCore.SIGNAL('editingFinished()'))

    def _browseClicked(self):
        if not self._browseFunc:
            return # no browse function defined
        result = self._browseFunc(self.lineEdit.text())
        if result != None:
            self._textBrowsed(result)

    def text(self):
        "Return the text in the text field"
        return self.lineEdit.text()

    def setText(self, s):
        "Set the text in the text field"
        self.lineEdit.setText(s)

class TextAndBrowseGroupBox(QtGui.QGroupBox):
    """
    A widget that has both a text input field and a browse button
    """
    def __init__(self, label='', browseLabel='', browseFunc=None, default='', browseIcon=None, parent=None):
        """
        Instantiate a TextAndBrowseWidget

        label is the name of the field
        browseLabel is the label on the browse button
        browseFunc is the function called when the browse
                   button is pressed and either returns
                   None for no change or returns a new string
                   It takes the string in the text field as input
        default is the default value in the text field
        browseIcon is an optional icon for the button

        This widget emits the following signals:
          textChanged(QString) - when the text changes
          textEdited(QString) - when the text has been edited in the edit box
          textBrowsed(QString) - when the text has been changed by the browser

        This widget accepts the following slots:
          setText(str) - set the text in the input field
        """
        QtGui.QGroupBox.__init__(self, label, parent)

        # create the UI
        self.lineEdit = QtGui.QLineEdit(default)
        if browseIcon:
            self.browseButton = QtGui.QPushButton(browseIcon,browseLabel)
        else:
            self.browseButton = QtGui.QPushButton(browseLabel)
        self._browseFunc = browseFunc
        self.groupLayout = QtGui.QHBoxLayout()
        self.groupLayout.addWidget(self.lineEdit)
        self.groupLayout.addWidget(self.browseButton)
        self.groupLayout.setStretchFactor(self.lineEdit,1)
        self.setLayout(self.groupLayout)

        # connect the signals
        self.connect(self.lineEdit,QtCore.SIGNAL('textChanged(QString)'),
                     self._textChanged)
        self.connect(self.lineEdit,QtCore.SIGNAL('textEdited(QString)'),
                     self._textEdited)
        self.connect(self.browseButton,QtCore.SIGNAL('clicked(bool)'),
                     self._browseClicked)

    def _textChanged(self, s):
        self.emit(QtCore.SIGNAL('textChanged(QString)'), s)

    def _textEdited(self, s):
        self.emit(QtCore.SIGNAL('textEdited(QString)'), s)

    def _textBrowsed(self, s):
        self.emit(QtCore.SIGNAL('textBrowsed(QString)'), s)

    def _browseClicked(self):
        if not self._browseFunc:
            return # no browse function defined
        result = self._browseFunc(self.lineEdit.text())
        if result != None:
            self.lineEdit.setText(result)
            self._textBrowsed(result)

    def text(self):
        "Return the text in the text field"
        return self.lineEdit.text()

    def setText(self, s):
        "Set the text in the text field"
        self.lineEdit.setText(s)

class PathSelectorWidget(TextAndBrowseWidget):
    def __init__(self,  browseCaption='', default='', parent=None):
        TextAndBrowseWidget.__init__(self, 'Browse...', self.browseFunc, default, None, parent)
        self.browseCaption = browseCaption

    def browseFunc(self, default):
        result = QtGui.QFileDialog.getExistingDirectory(self, self.browseCaption, default)
        if result:
            return result
        else:
            return None

class FileSaveSelectorWidget(TextAndBrowseWidget):
    def __init__(self, label='Save to', browseCaption='', defaultFile='', defaultPath='', fileFilter='All files (*.*)', parent=None):
        TextAndBrowseWidget.__init__(self, label, 'Browse...', self.browseFunc, defaultFile, None, parent)
        self.browseCaption = browseCaption
        self.filter = fileFilter
        self.defaultPath = defaultPath

    def browseFunc(self, default):
        result = str(QtGui.QFileDialog.getSaveFileName(self, self.browseCaption, os.path.join(str(self.defaultPath),str(default)), self.filter))
        if result:
            path, filename = os.path.split(result)
            if not os.path.samefile(path,self.defaultPath):
                self.defaultPath = path
                self._pathChanged(path)
            return QtCore.QString(filename)
        else:
            return None

    def _pathChanged(self, s):
        self.emit(QtCore.SIGNAL('pathChanged(QString)'), QtCore.QString(s))

class CustomOptionSelectorWidget(QtGui.QWidget):
    def __init__(self,
                 caption='Options:',
                 options=[('Option 1','Value 1'),
                          ('Option 2','Value 2')],
                 custom='Custom',
                 customConvert=str,
                 show=True,
                 parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.options = list(options)

        self.layout = QtGui.QHBoxLayout()
        self.layout.setMargin(0)
        self.custom = custom
        self.customConvert = customConvert

        if caption:
            self.label = QtGui.QLabel(caption, self)
            self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.layout.addWidget(self.label)
            self.layout.setStretchFactor(self.label,1)
        
        self.selector = QtGui.QComboBox(self)
        for key,value in self.options:
            self.selector.addItem(key)

        self.layout.addWidget(self.selector)

        if self.options:
            self.field = QtGui.QLineEdit(str(self.options[0][1]), self)
        else:
            self.field = QtGui.QLineEdit('', self)

        if show:
            self.layout.addWidget(self.field)
        else:
            self.field.hide()

        if self.custom:
            self.selector.addItem(custom)
        else:
            self.field.setReadOnly(True)

        self.setLayout(self.layout)

        self.connect(self.selector, QtCore.SIGNAL('currentIndexChanged(int)'),
                     self.setIndex)
        if show:
            self.connect(self.field, QtCore.SIGNAL('textEdited(QString)'),
                         self.setValue)

    def setIndex(self, newIndex):
        "Set the proper text in the field"
        if newIndex < len(self.options):
            # a built in option
            if self.field.text() != str(self.options[newIndex][1]):
                self.field.setText(str(self.options[newIndex][1]))
                self.emitValueChanged()

    def setValue(self, value):
        "Set the proper option in the pull-down box"
        values = [x[1] for x in self.options]
        try:
            selectedIndex = values.index(value)
            self.selector.setCurrentIndex(selectedIndex)
        except ValueError:
            if self.custom:
                self.selector.setCurrentIndex(len(self.options))
                self.field.setText(str(value))
            else:
                raise Error('Option %s is not a selectable option and no custom option is allowed' % s)
        self.emitValueChanged()

    def emitValueChanged(self):
        "Emit a valueChanged signal"
        self.emit(QtCore.SIGNAL('valueChanged()'))

    def value(self):
        "Return the value of the text field"
        try:
            return self.options[self.selector.currentIndex()][1]
        except IndexError:
            if self.custom:
                return self.customConvert(self.field.text())
            else:
                raise Error('Option %s is not a selectable option and no custom option is allowed' % self.field.text())

class TwinInfiniteSliderWidget(QtGui.QGroupBox):
    def __init__(self, coarseTick, fineTick, majorTickRatio=5, (strToValue, valueToStr)=(float, str), defaultValue=0.0, label='Slider', prefix='', suffix='', digits=2, eps=1e-4, directions='', parent=None):
        """
        Instantiate a slider system

        label - the name of the slider system 
        coarseTick - how much a minor tick represents on the coarse slider
        fineTick - how much a minor tick represents on the fine slider
        majorTickRatio - how many minor ticks per major tick 
        strToValue - a function that converts a string to a value
        valueToStr - a function that converts a value to a string
        prefix - a string to be prepended to the value string automatically
        suffix - a string to be appended to the value string automatically
        """
        QtGui.QGroupBox.__init__(self, label, parent)
        # save settings
        self.prefix = prefix
        self.suffix = suffix
        self.curValue = defaultValue
        self.digits = digits
        self.eps = eps
        self.directions = directions

        self.coarseTick = coarseTick
        self.fineTick = fineTick
        self.majorTickRatio = majorTickRatio
        self.strToValue = strToValue
        self.valueToStr = valueToStr

        # set up different backgrounds
        self.defaultPalette = QtGui.QPalette()
        self.errorPalette = QtGui.QPalette()
        self.errorPalette.setColor(QtGui.QPalette.Base,QtCore.Qt.red)
        self.editPalette = QtGui.QPalette()

        # create the gui
        self.tickIndicator = controls.Tick('s')
        
        self.coarseLabel = QtGui.QLabel('Coarse: ')
        self.coarseLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.coarseSlider = controls.InfiniteScroller(QtCore.Qt.Horizontal)
        self.coarseSlider.setThickness(45)
        self.coarseSlider.setMinorTickInterval(5)
        self.coarseSlider.setMajorTickInterval(self.majorTickRatio * self.coarseSlider.minorTickInterval())
        self.coarseSlider.setScale(1.0*self.coarseTick/self.coarseSlider.minorTickInterval())
        self.coarseTip = 'Drag the slider or click on it and use arrow keys for coarse adjustments'
        self.coarseSlider.setToolTip(self.coarseTip)
        self.fineLabel = QtGui.QLabel('Fine: ')
        self.fineLabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.fineSlider = controls.InfiniteScroller(QtCore.Qt.Horizontal)
        self.fineSlider.setThickness(30)
        self.fineSlider.setMinorTickInterval(5)
        self.fineSlider.setMajorTickInterval(self.majorTickRatio * self.fineSlider.minorTickInterval())
        self.fineSlider.setScale(1.0*self.fineTick/self.fineSlider.minorTickInterval())
        self.fineTip='Drag the slider or click on it and use arrow keys for fine adjustments'
        self.fineSlider.setToolTip(self.fineTip)
        self.fineSlider.setLabelFunction(None)

        self.prefixLabel = QtGui.QLabel(self.prefix)
        self.curEditor = components.FloatEditor(None, self.curValue, digits=self.digits, sig=False)
        self.suffixLabel = QtGui.QLabel(self.suffix)

        self.directionsLabel = QtGui.QLabel(self.directions)
        self.directionsLabel.setWordWrap(True)
        
        self.groupLayout = QtGui.QGridLayout()
        self.groupLayout.setVerticalSpacing(0)
        self.groupLayout.addWidget(self.tickIndicator,0,1)
        self.groupLayout.addWidget(self.coarseLabel,1,0)
        self.groupLayout.addWidget(self.coarseSlider,1,1)
        self.groupLayout.addWidget(self.fineLabel,2,0)
        self.groupLayout.addWidget(self.fineSlider,2,1)
        self.groupLayout.addWidget(self.prefixLabel,1,2)
        self.groupLayout.addWidget(self.curEditor,1,3)
        self.groupLayout.addWidget(self.suffixLabel,1,4)
        self.groupLayout.addWidget(self.directionsLabel,3,0,1,5)
        self.groupLayout.setSpacing(0)
        self.groupLayout.setColumnStretch(1,10)
        self.groupLayout.setColumnStretch(3,3)
        self.setLayout(self.groupLayout)

        # connect the signals
        self.connect(self.curEditor,
                     QtCore.SIGNAL('valueChanged()'),
                     self.handleEditorChange)
        self.connect(self.fineSlider, QtCore.SIGNAL('valueChanged(float)'),
                     self.handleSliderChange)
        self.connect(self.coarseSlider, QtCore.SIGNAL('valueChanged(float)'),
                     self.handleSliderChange)
        
    def handleEditorChange(self):
        value = self.curEditor.value()
        if value != self.value():
            self.setValue(value)
            self.fineSlider.setValue(value)
            self.coarseSlider.setValue(value)

    def handleSliderChange(self, value):
        if value != self.value():
            self.setValue(value)
            self.curEditor.setValue(value)
            self.fineSlider.setValue(value)
            self.coarseSlider.setValue(value)

    def setValue(self, value):
        if abs(value) < self.eps:
            value = 0
        if self.curValue != value:
            self.curValue = value
            self.curEditor.setValue(value)
            self.fineSlider.setValue(value)
            self.coarseSlider.setValue(value)
            self.emit(QtCore.SIGNAL('valueChanged()'))

    def value(self):
        return self.curValue

    def setDirections(self, directions):
        self.directions = directions
        self.directionsLabel.setText(self.directions)

    def setToolTip(self, toolTip):
        if toolTip:
            self.coarseSlider.setToolTip('')
            self.fineSlider.setToolTip('')
        else:
            self.coarseSlider.setToolTip(self.coarseTip)
            self.fineSlider.setToolTip(self.fineTip)
        QtGui.QGroupBox.setToolTip(self, toolTip)

class ChoiceDialog(QtGui.QDialog):
    def __init__(self, title, choices, rememberText='Remember my choice', parent=None):
        QtGui.QDialog.__init__(self, parent)
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle(title)
        self.buttons = []
        self.mapper = QtCore.QSignalMapper(self)
        i = 1
        # create buttons and connect to signal mapper, which
        # will determine where signals came from and pass the
        # appropriate argument into the signal it emits
        for choice in choices:
            self.buttons.append(QtGui.QPushButton(choice))
            layout.addWidget(self.buttons[-1])
            self.mapper.setMapping(self.buttons[-1], i)
            self.connect(self.buttons[-1], QtCore.SIGNAL('clicked(bool)'),
                         self.mapper, QtCore.SLOT('map()'))
            i=i+1
        self.connect(self.mapper, QtCore.SIGNAL('mapped(int)'),
                     self.done)
        if rememberText:
            self.remember = QtGui.QCheckBox(rememberText)
            layout.addWidget(self.remember)
        else:
            self.remember = QtGui.QCheckBox('') # not displayed
            self.remember.setVisible(False)

