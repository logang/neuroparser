"""
Some GUI control widgets
"""

from PyQt4 import QtCore, QtGui

import dragger

class Error(Exception):
    pass

class InfiniteScroller(QtGui.QFrame):
    """
    A infinite scrolling bar, with ridge marks
    """
    def __init__(self, orientation=QtCore.Qt.Horizontal, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self._majorTickInterval = 16
        self._minorTickInterval = 4
        self._scale = 1.0 # how much does each pixel on screen correspond to
        self._orientation = orientation
        self._minimum = None
        self._maximum = None
        self._value = 0.0
        self._length = 128 # includes border
        self._thickness = 16 # includes border
        self._labelFunction = lambda x: str(int(round(x)))
        
        self.setFrameStyle(self.StyledPanel | self.Sunken)
        if orientation == QtCore.Qt.Horizontal:
            self.setSizePolicy(QtGui.QSizePolicy.Preferred,QtGui.QSizePolicy.Fixed)
        else:
            self.setSizePolicy(QtGui.QSizePolicy.Fixed,QtGui.QSizePolicy.Preferred)        
            
        controlset = {
            ((),(QtCore.Qt.LeftButton,)) : self.drag,
        }

        self.controls = {'default':controlset}

        # set up the drag helper
        self.dragger = dragger.Dragger(self.mousePosition,
                                       self.value,
                                       float,
                                       lambda : 'default',
                                       self.controls,
                                       self)

        # set our focus policy
        self.setFocusPolicy(QtCore.Qt.ClickFocus)

    # property getters

    def length(self):
        return self._length

    def majorTickInterval(self):
        return self._majorTickInterval

    def maximum(self):
        return self._maximum

    def minimum(self):
        return self._minimum

    def minorTickInterval(self):
        return self._minorTickInterval

    def orientation(self):
        return self._orientation

    def scale(self):
        return self._scale

    def thickness(self):
        return self._thickness

    def labelFunction(self):
        return self._labelFunction

    def value(self):
        return self._value

    # property setters

    def setLength(self, length):
        self._length = length
        self.updateGeometry()
        self.update()

    def setMajorTickInterval(self, value):
        self._majorTickInterval = value
        self.update()

    def setMaximum(self, value):
        if value < self._minimum:
            value = self._minimum
        self._maximum = value
        self.validate()

    def setMinimum(self, value):
        if value > self._maximum:
            value = self._maximum
        self._minimum = value
        self.validate()

    def setMinorTickInterval(self, value):
        self._minorTickInterval = value
        self.update()

    def setOrientation(self, value):
        self._orientation = value
        self.updateGeometry()
        self.update()

    def setScale(self, value):
        if value == 0.0:
            raise Error('Illegal scale value, must be in value per pixel')
        self._scale = value

    def setThickness(self, value):
        self._thickness = value
        self.updateGeometry()
        self.update()

    def setLabelFunction(self, foo):
        self._labelFunction = foo
        self.update()

    def setValue(self, value):
        changed = (value != self._value)
        old_value = self._value
        self._value = value
        if changed:
            if self.validate():
                self.emit(QtCore.SIGNAL('valueChanged(float)'), self._value)
                self.emit(QtCore.SIGNAL('valueMoved(float)'), self._value-old_value)
            self.update()

    def moveValue(self, amount):
        self.setValue(self._value + amount)

    # geometry

    def sizeHint(self):
        if self._orientation == QtCore.Qt.Horizontal:
            return QtCore.QSize(self._length,
                                self._thickness)
        else:
            return QtCore.QSize(self._thickness,
                                self._length)

    # display

    def paintEvent(self, event):

        painter = QtGui.QPainter(self)
        palette = QtGui.QPalette()
        style = self.style()

        # get the size we want
        size = self.size()

        # clip to inside frame
        rect = QtCore.QRect(self.frameWidth(),self.frameWidth(),
                            size.width()-2*self.frameWidth(),
                            size.height()-2*self.frameWidth())
        painter.setClipRect(rect)

        # draw the background
        if self.hasFocus():
            # selected background
            background = palette.highlight()
        else:
            # unselected background
            background = palette.button()

        painter.fillRect(rect, background)

        # draw the major ticks
        lightPen = QtGui.QPen(palette.light(), 1)
        shadowPen = QtGui.QPen(palette.dark(), 1)
        textPen = QtGui.QPen(palette.buttonText(), 1)
        highlightTextPen = QtGui.QPen(palette.highlightedText(), 1)

        int_value = int(round(self._value / self._scale))

        if self._minorTickInterval > 0:
            start = int(round(self._thickness*0.25))
            end = self._thickness-start-1
            if self._orientation == QtCore.Qt.Horizontal:
                offset1 = (int_value + (size.width()-1)/2) % self._minorTickInterval
                offset2 = (int_value + 1 + (size.width()-1)/2) % self._minorTickInterval
                painter.setPen(lightPen)
                for x in range(offset1,size.width(),self._minorTickInterval):
                    painter.drawLine(x, start, x, end)
                painter.setPen(shadowPen)
                for x in range(offset2,size.width(),self._minorTickInterval):
                    painter.drawLine(x, start, x, end)
            else:
                offset1 = (int_value + (size.height()-1)/2) % self._minorTickInterval
                offset2 = (int_value + 1 + (size.height()-1)/2) % self._minorTickInterval
                painter.setPen(lightPen)
                for y in range(offset1,size.height(),self._minorTickInterval):
                    painter.drawLine(start, y, end, y)
                painter.setPen(shadowPen)
                for y in range(offset2,size.height(),self._minorTickInterval):
                    painter.drawLine(start, y, end, y)
        
        if self._majorTickInterval > 0:
            if self._orientation == QtCore.Qt.Horizontal:
                offset1 = (int_value + (size.width()-1)/2) % self._majorTickInterval
                offset2 = (int_value + 1 + (size.width()-1)/2) % self._majorTickInterval
                painter.setPen(lightPen)
                for x in range(offset1,size.width(),self._majorTickInterval):
                    painter.drawLine(x, 0, x, self._thickness-1)
                painter.setPen(shadowPen)
                for x in range(offset2,size.width(),self._majorTickInterval):
                    painter.drawLine(x, 0, x, self._thickness-1)
                # draw labels
                if self._labelFunction:
                    painter.setBackgroundMode(QtCore.Qt.OpaqueMode)
                    if self.hasFocus():
                        painter.setPen(highlightTextPen)
                        painter.setBackground(palette.highlight())
                    else:
                        painter.setPen(textPen)
                        painter.setBackground(palette.button())
                    for x in range(offset1,size.width(),self._majorTickInterval):
                        value = ((size.width()-1)/2 - x + int_value)*self._scale 
                        label = ' '+self._labelFunction(value)+' '
                        pos_x = x - self._majorTickInterval/2
                        pos_y = self._thickness/4
                        painter.drawText(pos_x,pos_y,
                                         self._majorTickInterval, self._thickness/2,
                                         QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                                         label)
                    painter.setBackgroundMode(QtCore.Qt.TransparentMode)
            else:
                offset1 = (int_value + (size.height()-1)/2) % self._majorTickInterval
                offset2 = (int_value + 1 + (size.height()-1)/2) % self._majorTickInterval
                painter.setPen(lightPen)
                for y in range(offset1,size.height(),self._majorTickInterval):
                    painter.drawLine(0, y, self._thickness-1, y)
                painter.setPen(shadowPen)
                for y in range(offset2,size.height(),self._majorTickInterval):
                    painter.drawLine(0, y, self._thickness-1, y)
                if self._labelFunction:
                    painter.setBackgroundMode(QtCore.Qt.OpaqueMode)
                    if self.hasFocus():
                        painter.setPen(highlightTextPen)
                        painter.setBackground(palette.highlight())
                    else:
                        painter.setPen(textPen)
                        painter.setBackground(palette.button())
                    for y in range(offset1,size.height(),self._majorTickInterval):
                        value = ((size.height()-1)/2 - y + int_value)*self._scale 
                        label = ' '+self._labelFunction(value)+' '
                        pos_y = y - self._majorTickInterval/2
                        pos_x = self._thickness/4
                        painter.drawText(pos_x,pos_y,
                                         self._thickness/2, self._majorTickInterval,
                                         QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter,
                                         label)
                    painter.setBackgroundMode(QtCore.Qt.TransparentMode)


        # paint the border
        rect = QtCore.QRect(0,0,size.width(),size.height())
        self.setFrameRect(rect)
        painter = None
        QtGui.QFrame.paintEvent(self, event)

    # validation

    def validate(self):
        """
        Clamp the value within bounds, if necessary, and return True
        if the original value did not need to be changed to conform
        """
        changed = False
        old_value = self._value
        if self._maximum is not None and self._value > self._maximum:
            self._value = self._maximum
            changed = True
        if self._minimum is not None and self._value < self._minimum:
            self._value = self._minimum
            changed = True
        if changed:
            self.emit(QtCore.SIGNAL('valueChanged(float)'), self._value)
            self.emit(QtCore.SIGNAL('valueMoved(float)'), self._value - old_value)
        return not changed

    # interaction

    def mousePosition(self):
        "Return the current mouse position"
        pos = self.mapFromGlobal(QtGui.QCursor.pos())
        return pos.x(), pos.y()

    def mousePressEvent(self, event):
        "Pass a mouse press event to the drag handler"
        self.dragger.mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        "Pass a mouse release event to the drag handler"
        self.dragger.mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        "Pass a mouse move event to the drag handler"
        self.dragger.mouseMoveEvent(event)

    def keyPressEvent(self, event):
        """
        Pass a key press event to the drag handler, but ignore
        it to propagate it up as well, since the key only
        affects state and does not trigger events
        """
        if (event.key() == QtCore.Qt.Key_Left or
            event.key() == QtCore.Qt.Key_Down):
            delta = -self._scale
            self.setValue(self.value()+delta)
        elif (event.key() == QtCore.Qt.Key_Right or
              event.key() == QtCore.Qt.Key_Up):
            delta = self._scale
            self.setValue(self.value()+delta)
        elif (event.key() == QtCore.Qt.Key_Home or
              event.key() == QtCore.Qt.Key_PageDown):
            delta = -10*self._scale
            self.setValue(self.value()+delta)
        elif (event.key() == QtCore.Qt.Key_End or
              event.key() == QtCore.Qt.Key_PageUp):
            delta = 10*self._scale
            self.setValue(self.value()+delta)
        else:
            self.dragger.keyPressEvent(event)
            event.ignore()

    def keyReleaseEvent(self, event):
        """
        Pass a key release event to the drag handler, but ignore
        it to propagate it up as well, since the key only
        affects state and does not trigger events
        """
        self.dragger.keyReleaseEvent(event)
        event.ignore()

    def drag(self, event, state, (dx, dy, x, y), keys, buttons):
        """
        Handle dragging the slider
        """
        if self._orientation == QtCore.Qt.Horizontal:
            delta = dx*self._scale
        else:
            delta = dy*self._scale
        self.setValue(state+delta)

class Tick(QtGui.QWidget):
    def __init__(self, direction='S', parent=None):
        QtGui.QWidget.__init__(self, parent)

        self._length = 128 # includes border
        self._thickness = 16 # includes border
        self._direction = direction

        if self._direction.lower() in 'ns':
            self.setSizePolicy(QtGui.QSizePolicy.Preferred,
                               QtGui.QSizePolicy.Fixed)
        elif self._direction.lower() in 'ew':
            self.setSizePolicy(QtGui.QSizePolicy.Fixed,
                               QtGui.QSizePolicy.Preferred)

            
    # property getters

    def length(self):
        return self._length

    def direction(self):
        """
        Reveals which way the tick is pointing
        """
        return self._direction

    def thickness(self):
        return self._thickness

    # property setters

    def setLength(self, length):
        self._length = length
        self.updateGeometry()
        self.update()

    def setDirection(self, value):
        self._direction = value
        self.updateGeometry()
        self.update()

    def setThickness(self, value):
        self._thickness = value
        self.updateGeometry()
        self.update()

    # geometry

    def sizeHint(self):
        if self._direction.lower() in 'ns':
            return QtCore.QSize(self._length,
                                self._thickness)
        else:
            return QtCore.QSize(self._thickness,
                                self._length)

    # display

    def paintEvent(self, event):

        painter = QtGui.QPainter(self)
        palette = QtGui.QPalette()
        style = self.style()

        # get the size we want
        size = self.size()

        # clip to inside frame
        rect = QtCore.QRect(0,0,size.width(),size.height())
        painter.setClipRect(rect)

        # draw the tick
        if self._direction in 'ns':
            length = size.width()
        else:
            length = size.height()
        centerPos = (length) / 2
        centerWidth = 16
        centerLength = self._thickness

        brush1 = QtGui.QBrush(palette.light())
        brush2 = QtGui.QBrush(palette.dark())
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))

        if self._direction == 'n':
            # points up
            points = QtGui.QPolygon(3)
            painter.setBrush(brush1)
            points.setPoint(0, centerPos, 0)
            points.setPoint(1, centerPos+centerWidth/2, centerLength)
            points.setPoint(2, centerPos-centerWidth/2, centerLength)
            painter.drawPolygon(points)
            painter.setBrush(brush2)
            points.setPoint(0, centerPos+1, 0)
            points.setPoint(1, centerPos+centerWidth/2+1, centerLength)
            points.setPoint(2, centerPos-centerWidth/2+1, centerLength)
            painter.drawPolygon(points)
        elif self._direction == 's':
            # points up
            points = QtGui.QPolygon(3)
            painter.setBrush(brush1)
            points.setPoint(0, centerPos, centerLength)
            points.setPoint(1, centerPos+centerWidth/2, 0)
            points.setPoint(2, centerPos-centerWidth/2, 0)
            painter.drawPolygon(points)
            painter.setBrush(brush2)
            points.setPoint(0, centerPos+1, centerLength)
            points.setPoint(1, centerPos+centerWidth/2+1, 0)
            points.setPoint(2, centerPos-centerWidth/2+1, 0)
            painter.drawPolygon(points)
        painter = None
        QtGui.QWidget.paintEvent(self, event)

