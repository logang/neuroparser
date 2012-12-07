"""
Dragger - a QObject to handle mouse dragging
"""

from PyQt4 import QtCore, QtGui

import threading

def _viable((keys, buttons), actual_keys, actual_buttons):
    "Determine whether a certain combo fits the current state"
    for key in keys:
        if key not in actual_keys:
            return False
    for button in buttons:
        if button not in actual_buttons:
            return False
    return True

class Dragger(QtCore.QObject):
    def __init__(self, posProvider, stateProvider, stateCopier,
                 modeProvider, controls, parent=None, lru=False):
        """
        Initialize the DragStack

        posProvider - the function that returns a copy of the current mouse position
        stateProvider - the function that returns a copy of the current state
        stateCopier - a function that returns a copy of the state given as input
        modeProvider - a function that returns the current active control set
        controls - set of controls for the dragger, indexed by a string denoting the current mode
                   each control set is a dictionary mapping (keys,buttons) to a function
                   keys and buttons are tuples of Qt keys and buttons corresponding to the
                   drag action
                   a mouse drag event triggers a callback on the function
                   which expects: 1) the original mouse event
                                  2) the state at the start of a drag
                                  3) a tuple (dx, dy, x, y)
                                     where x and y are positions at the start of a drag
                                  4) a list of keyboard buttons
                                  5) a list of mouse buttons
        
        parent - the QObject that owns this dragger
        lru - whether older mouse clicks have precedence
        """
        QtCore.QObject.__init__(self, parent)

        self.lru = lru
        
        self.lock = threading.RLock()

        self.startPos = None # the mouse position at the start of a drag
        self.startState = None # state at the start of a drag
        self.posProvider = posProvider
        self.stateProvider = stateProvider
        self.stateCopier = stateCopier
        self.modeProvider = modeProvider
        self.controls = controls

        # initialize keyboard and mouse state
        self.buttons = []
        self.keys = []

    def clear(self):
        """
        Reset the dragging
        """
        self.lock.acquire()
        self.startPos = None
        self.startState = None
        self.lock.release()

    def update(self):
        """
        Update the state
        """
        self.lock.acquire()
        self.startPos = self.posProvider()
        self.startState = self.stateProvider()
        self.lock.release()

    def mousePressEvent(self, event):
        """
        Add the mouse button to the button state
        """
        self.lock.acquire()
        while event.button() in self.buttons:
            self.buttons.remove(event.button())
        if self.lru:
            self.buttons.append(event.button())
        else:
            self.buttons.insert(0,event.button())
        self.lock.release()
        self.update()

    def mouseReleaseEvent(self, event):
        """
        Remove the mouse button from the button state
        """
        self.lock.acquire()
        while event.button() in self.buttons:
            self.buttons.remove(event.button())
        self.lock.release()
        self.update()

    def keyPressEvent(self, event):
        """
        Add the key to the key state
        """
        self.lock.acquire()
        while event.key() in self.keys:
            self.keys.remove(event.key())
        self.keys.append(event.key())
        self.lock.release()
        self.update()

    def keyReleaseEvent(self, event):
        """
        Remove the key from the key state
        """
        self.lock.acquire()
        while event.key() in self.keys:
            self.keys.remove(event.key())
        self.lock.release()
        self.update()

    def mouseMoveEvent(self, event):
        """
        Calculate the dx and dy of a mouse drag/move and
        call the callback function
        """
        callback = False
        self.lock.acquire()
        try:
            if self.buttons or self.keys:
                x, y = self.startPos
                curx, cury = self.posProvider()
                dx, dy = curx-x,cury-y
                state = self.stateCopier(self.startState)
                keys, buttons = tuple(self.keys), tuple(self.buttons)
                callback = True
        finally:
            self.lock.release()
        if callback:
            # grab the current mode
            mode = self.modeProvider()
            if mode not in self.controls:
                # unsupported manipulation mode
                event.ignore()
                raise Error('Unsupported manipulation mode: '+mode)
            control_set = self.controls[mode]

            # find a fit for the operation

            # trim the playing field
            possible_combos = [_x for _x in control_set.keys() if _viable(_x,keys,buttons)]

            if len(possible_combos) == 0:
                # no matching combo
                event.ignore()
                return
            if len(possible_combos) == 1:
                # only one match, so this must be it
                control_set[possible_combos[0]](event, state, (dx, dy, x, y),
                                                keys, buttons)
                return

            # find the best match using mouse buttons
            # this assumes that the list being iterated has latest addition first
            # and that iteration is forwards
            for button in buttons:
                temp_list = [(_keys,_buttons) for (_keys,_buttons) in possible_combos if
                             button in _buttons]
                if temp_list: # if we can trim it down without killing it
                    possible_combos = temp_list
                    # see if we trimmed it down enough
                    if len(possible_combos) == 1:
                        # only one match, so this must be it
                        control_set[possible_combos[0]](event, state, (dx, dy, x, y),
                                                        keys, buttons)
                        return

            # find the best match using keys
            # see above for notes
            for key in keys:
                temp_list = [(_keys,_buttons) for (_keys,_buttons) in possible_combos if
                             key in _keys]
                if temp_list: # if we can trim it down without killing it
                    possible_combos = temp_list
                    # see if we trimmed it down enough
                    if len(possible_combos) == 1:
                        # only one match, so this must be it
                        control_set[possible_combos[0]](event, state, (dx, dy, x, y),
                                                        keys, buttons)
                        return

            # we should not get here
            raise Error('Unexpected lack of key/button match')
        else:
            event.ignore()
