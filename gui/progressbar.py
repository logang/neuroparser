from PyQt4 import QtCore, QtGui, Qt

class LedProgressBar (QtGui.QWidget):

    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)

        self.minVal = 0
        self.maxVal = 100
        self.currentVal = 0
        self.fontdim = 30
        self.lengthbar = 0
        self.colBar = QtCore.Qt.green

    def paintEvent(self, event):
        self.paintBorder()
        self.paintBar()
        self.paintLine()
        self.paintValue()

    def paintBorder(self):
        painter = QtGui.QPainter(self)
        painter.setWindow(0, 0, 470, 80)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        linGrad = QtGui.QLinearGradient(5, 40, 15, 40)
        linGrad.setColorAt(0, QtCore.Qt.white)
        linGrad.setColorAt(1, QtCore.Qt.black)
        linGrad.setSpread(QtGui.QGradient.PadSpread)
        painter.setBrush(linGrad)
        border = QtCore.QRectF(5, 5, 460, 70)
        painter.drawRoundRect(border, 3)

        painter.setBrush(QtGui.QColor(70, 70, 70));
        value = QtCore.QRectF(385, 10, 75, 60);
        painter.drawRoundRect(value, 15);

    def paintBar(self):
        painter = QtGui.QPainter(self);
        painter.setWindow(0, 0, 470, 80);
        painter.setRenderHint(QtGui.QPainter.Antialiasing);
        painter.setBrush(QtGui.QColor(70, 70, 70));

        # background color
        back = QtCore.QRectF(20, 10, 360, 60);
        painter.drawRoundRect(back, 3);

        if (self.currentVal > self.maxVal):
            return
        if (self.currentVal < self.minVal):
            return

        # waiting state if min = max
        if(self.minVal == self.maxVal):
            painter.setBrush(self.colBar);
            bar = QtCore.QRectF(40, 10, 40, 60);
            bar1 = QtCore.QRectF(130, 10, 40, 60);
            bar2 = QtCore.QRectF(220, 10, 40, 60);
            bar3 = QtCore.QRectF(310, 10, 40, 60);
            painter.drawRoundRect(bar, 3);
            painter.drawRoundRect(bar1, 3);
            painter.drawRoundRect(bar2, 3);
            painter.drawRoundRect(bar3, 3);
            return

        # check positive or negative scale
        if (self.maxVal >= 0 and self.minVal >= 0 or self.maxVal >= 0 and self.minVal <= 0):
            self.lengthBar = 360-360 * (self.maxVal-self.currentVal)/(self.maxVal-self.minVal)
        if (self.maxVal <= 0 and self.minVal <= 0):
            self.lengthBar = 360 * (self.minVal-self.currentVal)/(self.minVal-self.maxVal);

        # length and color bar
        painter.setBrush(self.colBar);
        bar = QtCore.QRectF(20, 10, self.lengthBar, 60);
        painter.drawRoundRect(bar, 3);
        #        self.emit(QtCore.SIGNAL('valueChanged(int)', self.currentVal));

    def paintLine(self):
        painter = QtGui.QPainter(self)
        painter.setWindow(0, 0, 470, 80);
        painter.setRenderHint(QtGui.QPainter.Antialiasing);
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine,
                                  QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin));

        for i in range(40):
            painter.drawLine(20+(360/40.0*i), 10, 20+(360/40.0*i), 70);

    def paintValue(self):
        painter = QtGui.QPainter(self)
        painter.setWindow(0, 0, 470, 80)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        value = QtCore.QRectF(385, 10, 75, 60)
        font = QtGui.QFont("Arial", self.fontdim, QtGui.QFont.Normal)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        x = 100.0
        y = 360.0
        val = "%0.0f%%" % (self.lengthBar*(x/y))
        painter.drawText(value, QtCore.Qt.AlignCenter, val);

    def setMinValue(self, val):
        self.minVal = val
        self.update()

    def setMaxValue(self, val):
        self.maxVal = val
        self.update()

    def setCurrentValue(self, val):
        self.currentVal = val
        self.update()

    def setFontDim(self, val):
        self.fontdim = val
        self.update()

    def setPrecision(self, val):
        self.precision = val
        self.update()

    def setBarColor(self, color):
        self.color = color
        self.update()

        #    def minimumSizeHint(self):
        #        return QtCore.QSize(10, 10)

    def sizeHint(self):
        return QtCore.QSize(150, 50);


