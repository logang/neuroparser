"Widget components"

from PyQt4 import QtCore, QtGui

class SlimGroupBox(QtGui.QGroupBox):
    "A slim group box"
    def __init__(self, title='', parent=None):
        QtGui.QGroupBox.__init__(self, title, parent)
    
        # reset the margins
        if title:
            height = self.fontMetrics().height()
            margins = (0, ((height+1)>>1), 0, 0)
        else:
            margins = (0,0,0,0)
            
        self.setContentsMargins(*margins)
        
class SlimGroupBoxV(SlimGroupBox):
    "A slim group box with an embedded QVBoxLayout"
    def __init__(self, title='', parent=None):
        SlimGroupBox.__init__(self, title, parent)

        self._layout = QtGui.QVBoxLayout()
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

class SlimGroupBoxH(SlimGroupBox):
    "A slim group box with an embedded QHBoxLayout"
    def __init__(self, title='', parent=None):
        SlimGroupBox.__init__(self, title, parent)

        self._layout = QtGui.QHBoxLayout()
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

class SlimGroupBoxG(SlimGroupBox):
    "A slim group box with an embedded QHBoxLayout"
    def __init__(self, title='', parent=None):
        SlimGroupBox.__init__(self, title, parent)

        self._layout = QtGui.QGridLayout()
        self._layout.setSpacing(0)
        self.setLayout(self._layout)

class FlexEditor(QtGui.QLineEdit):
    """
    A line editor that'll warn the user if there's an incorrect value

    It will emit valueChanged() signals for when the value changes
    """
    def __init__(self, verifyFunc,
                 strToValue, valueToStr,
                 defaultValue, parent=None):
        """
        Initialize the editor

        verifyFunc on a string returns 1 for valid, 0 for warning and -1 for invalid
        strToValue converts a string to a value
        valueToStr converts a value to a string
        defaultValue is the initial value
        """
        defaultStr = valueToStr(defaultValue)
        QtGui.QLineEdit.__init__(self, defaultStr)

        self.defaultPalette = QtGui.QPalette()
        self.errorPalette = QtGui.QPalette()
        self.errorPalette.setColor(QtGui.QPalette.Base, QtCore.Qt.red)
        self.warningPalette = QtGui.QPalette()
        self.warningPalette.setColor(QtGui.QPalette.Base, QtCore.Qt.yellow)

        if verifyFunc:
            self.verifyFunc = verifyFunc
        self.strToValue = strToValue
        self.valueToStr = valueToStr

        self._value = defaultValue

        metrics = self.fontMetrics()
        boundRect = metrics.boundingRect(defaultStr)
        self._ideal_width = int(boundRect.width()*1.5+0.5)
        self._ideal_height = boundRect.height()+4

        self.connect(self, QtCore.SIGNAL('textEdited(QString)'),
                     self.processTextEdit)

    def minimumSizeHint(self):
        return QtCore.QSize(self._ideal_width, self._ideal_height)

    def sizeHint(self):
        return QtCore.QSize(self._ideal_width, self._ideal_height)

    def value(self):
        return self._value

    def setValue(self, newValue, updateText=True):
        if newValue != self._value:
            self._value = newValue
            if updateText:
                newText = self.valueToStr(newValue)
                self.setText(newText)
            self.emit(QtCore.SIGNAL('valueChanged()'))

    def processTextEdit(self, s):
        text = self.text()
        validity = self.verifyFunc(text)
        if validity < 0:
            self.setPalette(self.errorPalette)
        elif validity == 0:
            self.setPalette(self.warningPalette)
        else:
            self.setPalette(self.defaultPalette)
            newValue = self.strToValue(text)
            self.setValue(newValue, False)

    def verifyFunc(self, s):
        "Dummy verification function"
        return 1

class FloatEditor(FlexEditor):
    """
    A floating point number editor
    """
    def __init__(self, verifyFunc, default=1.0, digits=4, sig=True, parent=None):
        """
        Initialize the floating point number editor

        verifyFunc is a function that takes a floating point number
                   and returns -1 for invalid, 0 for warning, and 1
                   for okay
        """
        self._digits = digits
        self._sig = sig
        FlexEditor.__init__(self, None, float, self.toString, default, parent)

        self.verifyFunc2 = verifyFunc
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    def toString(self, x):
        if self._sig:
            return ('%#.'+str(self._digits)+'g') % x
        else:
            return ('%.'+str(self._digits)+'f') % x

    def verifyFunc(self, s):
        try:
            a=float(s)
        except Exception:
            return -1
        if self.verifyFunc2:
            return self.verifyFunc2(a)
        else:
            return 1

class IntEditor(FlexEditor):
    """
    An integer number editor
    """
    def __init__(self, verifyFunc, default=1, parent=None):
        """
        Initialize the integer number editor

        verifyFunc is a function that takes an integer
                   and returns -1 for invalid, 0 for warning, and 1
                   for okay
        """
        FlexEditor.__init__(self, None, int, str, default, parent)

        self.verifyFunc2 = verifyFunc
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

    def verifyFunc(self, s):
        try:
            a=int(s)
        except Exception:
            return -1
        if self.verifyFunc2:
            return self.verifyFunc2(a)
        else:
            return 1
        
