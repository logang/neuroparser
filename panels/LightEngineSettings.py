"""
The Plugin Settings panel
"""
from PyQt4 import Qt, QtCore, QtGui, QtOpenGL
import serial
import liblo
from log import xenon_log, xenon_warning, xenon_error

from settings import Settings
import LightEnginePanel

# ---------------------------------------------------------------------------------

class LightEngineSettings(QtGui.QWidget, LightEnginePanel.Ui_lightEnginePanel):
    """
    A window that controls the SpectraX Lumencor light engine.
    """

    # Helper function
    def buttonStylesheet(self, button_name, button_on_color):
        return '''QPushButton#%s {
                   background-color: rgb(200, 200, 200);
                   border-style: outset;
                   border-width: 2px;
                   border-radius: 10px;
                   border-color: beige;
                   font: bold 14px;
                   width: 80px;
                   padding: 6px;
                  }
                  QPushButton#%s:on {
                   background-color: %s;
                   border-style: inset;
                  }
                  QPushButton#%s:selected {
                   border-color: black;
                   border-width: 4px;
                  }''' % (button_name, button_name, button_on_color, button_name)
        

    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)
        self.settings = Settings()

        # Populate the pull-down menu and select the previous plugin
        # we were using, if it is available.
        prev_port = None
        if self.settings['light_engine'].contains('comm_port'):
            prev_port = self.settings['light_engine'].comm_port

        self.current_port_idx = -1
        self.serial = None
        try:
            self.serial = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
            self.portComboBox.addItem(self.serial.portstr)
        except Exception, e:
            xenon_log('\t--> WARNING: Could not open light engine serial port:' + str(e))
        
        idx = self.portComboBox.findText(prev_port)
        if idx != -1:
            self.portComboBox.setCurrentIndex(idx)

        # Give each button a unique color and look
        self.violetButton.setStyleSheet(self.buttonStylesheet('violetButton', 'rgb(187, 0, 255)'))
        self.blueButton.setStyleSheet(self.buttonStylesheet('blueButton', 'rgb(0, 0, 255)'))
        self.cyanButton.setStyleSheet(self.buttonStylesheet('cyanButton', 'rgb(0, 255, 242)'))
        self.tealButton.setStyleSheet(self.buttonStylesheet('tealButton', 'rgb(0, 255, 170)'))
        self.greenButton.setStyleSheet(self.buttonStylesheet('greenButton', 'rgb(0, 255, 0)'))
        self.redButton.setStyleSheet(self.buttonStylesheet('redButton', 'rgb(255, 0, 0)'))

        # Connect the sliders and spinbox widgets, and deactivate them when the button is not enabled.
        self.connect(self.violetSlider, QtCore.SIGNAL('valueChanged(int)'), self.violetSpinBox.setValue)
        self.connect(self.violetSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.violetSlider.setValue)

        self.connect(self.blueSlider, QtCore.SIGNAL('valueChanged(int)'), self.blueSpinBox.setValue)
        self.connect(self.blueSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.blueSlider.setValue)

        self.connect(self.cyanSlider, QtCore.SIGNAL('valueChanged(int)'), self.cyanSpinBox.setValue)
        self.connect(self.cyanSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.cyanSlider.setValue)

        self.connect(self.tealSlider, QtCore.SIGNAL('valueChanged(int)'), self.tealSpinBox.setValue)
        self.connect(self.tealSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.tealSlider.setValue)

        self.connect(self.greenSlider, QtCore.SIGNAL('valueChanged(int)'), self.greenSpinBox.setValue)
        self.connect(self.greenSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.greenSlider.setValue)

        self.connect(self.redSlider, QtCore.SIGNAL('valueChanged(int)'), self.redSpinBox.setValue)
        self.connect(self.redSpinBox, QtCore.SIGNAL('valueChanged(double)'), self.redSlider.setValue)

        self.redState = False
        self.greenState = False
        self.cyanState = False
        self.violetState = False
        self.blueState = False
        self.tealState = False

        # Send initialization serial message
        self.serialInit()

        # Set default values
        self.violetButton.setOn(self.settings['light_engine'].refreshWithDefault('violetState', 0))
        self.blueButton.setOn(self.settings['light_engine'].refreshWithDefault('blueState', 0))
        self.cyanButton.setOn(self.settings['light_engine'].refreshWithDefault('cyanState', 0))
        self.tealButton.setOn(self.settings['light_engine'].refreshWithDefault('tealState', 0))
        self.greenButton.setOn(self.settings['light_engine'].refreshWithDefault('greenState', 0))
        self.redButton.setOn(self.settings['light_engine'].refreshWithDefault('redState', 0))
        self.greenYellowButton.setOn(self.settings['light_engine'].refreshWithDefault('greenYellowButton', 0))


        self.violetSlider.setValue(self.settings['light_engine'].refreshWithDefault('violetVal', 0))
        self.blueSlider.setValue(self.settings['light_engine'].refreshWithDefault('blueVal', 0))
        self.cyanSlider.setValue(self.settings['light_engine'].refreshWithDefault('cyanVal', 0))
        self.tealSlider.setValue(self.settings['light_engine'].refreshWithDefault('tealVal', 0))
        self.greenSlider.setValue(self.settings['light_engine'].refreshWithDefault('greenVal', 0))
        self.redSlider.setValue(self.settings['light_engine'].refreshWithDefault('redVal', 0))

        # Set up the OSC endpoints
        self.osc_client = None
        try:
            self.osc_client = liblo.Address('10.32.252.7', 9001)
        except liblo.AddressError, err:
            xenon_error("Could not create OSC client endpoint" + str(err))
            self.osc_client = None

        try:
            self.osc_server = OscServer(self, 9000)
        except liblo.ServerError, err:
            xenon_error("Could not create OSC server endpoint" + str(err))
            self.osc_server = None
        self.osc_server.start()

    def __del__(self):
        del self.osc_server
        del self.osc_client
        self.osc_client = None
        self.osc_server = None

    # --------------------------------------------------------------------
    #                         SERIAL ROUTINES
    #
    # These routines implement the serial interface for the
    # Lumencor SpectraX Light Engine (7 Channel Preliminary 11/20/09)
    #
    # This interface is available from the LumenCor website
    # --------------------------------------------------------------------

    # Sends bytes one at a time.  The bytes are specified as two
    # charater strings in a python list.
    def sendBytes(self, blist):
        # Only send bytes if the serial port is open.
        if not self.serial:
            xenon_log("Could not send serial byte.  The serial port is not open.", "light_engine")
            return
        bytestr = ''
        for byte in blist:
            if len(byte) > 2:
                raise Exception('Could not encode byte from string \'0x' + byte + '\'.  String was too short or too long.')
            bytestr += chr(int(byte,16))
        #        print 'SENDING: ', len(bytestr), blist
        self.serial.write(bytestr)
            

    # Initialization Command String for RS-232 Intensity and RS-232 OR
    # TTL Enables: The first two commands MUST be issued after every
    # power cycle to properly configure controls for further commands.
    def serialInit(self):
        self.sendBytes(['57', '02', 'FF', '50'])
        self.sendBytes(['57', '03', 'AB', '50'])

    # Note- If the Green Channel is enabled, then no other channels can
    # be enabled simultaneously. If other channels are enabled, then the
    # green channel enable will have priority.
    #
    def serialChannelEnable(self, red = False, green = False, cyan = False,
                            uv = False, blue = False, teal = False):
        byte_mask = int('0x7f', 16)
        if red:
            byte_mask ^= int('0x01', 16)
        if green:
            byte_mask ^= int('0x02', 16)
        if cyan:
            byte_mask ^= int('0x04', 16)
        if uv:
            byte_mask ^= int('0x08', 16)
        if blue:
            byte_mask ^= int('0x20', 16)
        if teal:
            byte_mask ^= int('0x40', 16)
        self.sendBytes(['4f', hex(byte_mask)[2:], '50'])

    # Select the Green / Yellow Filter setting
    def serialGreenYellowFilter(self, enable = False):
        if enable:
            self.sendBytes(['4f', '7D', '50'])
        else:
            self.sendBytes(['4f', '6D', '50'])

    # IIC DAC Intensity Control Command Strings
    #
    # Channel nums:
    #  0 - Red
    #  1 - Green
    #  2 - Cyan
    #  3 - UV
    #  4 - Blue
    #  5 - Teal
    #
    # Intensity is a integer between 0 and 255
    def serialChannelIntensity(self, channel, intensity):

        if (int(intensity) < 0 or int(intensity) > 255):
            raise Error('Intensity must be a value between 0 and 255')
        
        # Byte 5 is the DAC IIC Address. Red, Green, Cyan and UV use
        # IIC Addr = 18. Blue and Teal use IIC Addr = 1A.
        if channel == 4 or channel == 5:
            iic_address = '1A'
            if channel == 4:   # Blue
                dac_addr = '01'
            elif channel == 5: # Teal
                dac_addr = '02'            
        else:
            iic_address = '18'
            if channel == 0:   # Red
                dac_addr = '08'
            elif channel == 1: # Green
                dac_addr = '04'
            elif channel == 2: # Cyan
                dac_addr = '02'
            elif channel == 3: # UV
                dac_addr = '01'

        # Set the intensity bits.
        inverse_intensity = 255-int(intensity) 
        lower_bit = hex( (inverse_intensity & int('0x0F',16)) << 4)[2:]
        higher_bit = hex( ((inverse_intensity & int('0xF0',16)) >> 4) | int('0xF0',16))[2:]
        self.sendBytes(['53', iic_address, '03', dac_addr, higher_bit, lower_bit, '50'])

    # Reads the temperature of the lumencor, and returns it as a
    # floating point number in degrees celcius.
    def serialReadTemperature(self):

        # Issue the temperature read cmd.
        self.sendBytes(['53', '91', '02', '50'])
        x = self.serial.read(2)
        temperature = ((ord(x[0]) << 8 | ord(x[1])) >> 5) * 0.125
        return temperature
    
    # --------------------------------------------------------------------
    #                               ACTIONS 
    # --------------------------------------------------------------------

    def on_portComboBox_currentIndexChanged(self, index):
        # Ignore signals with string arguments
        if (type(index) is not int):
            return

        # Ignore if this is just the same argument
        if (index == self.current_port_idx):
            return

    # ----------------- BUTTONS -------------------

    def osc_send(self, path, val):
        pass
        # if self.osc_client:
        #     liblo.send(self.osc_client, path, val)

    def on_initButton_pressed(self):
        self.serialInit()

    def on_enableChannelsButton_pressed(self):
        self.violetButton.setOn(True)
        self.greenButton.setOn(True)
        self.blueButton.setOn(True)
        self.cyanButton.setOn(True)
        self.tealButton.setOn(True)
        self.redButton.setOn(True)

    def on_disableChannelsButton_pressed(self):
        self.violetButton.setOn(False)
        self.blueButton.setOn(False)
        self.cyanButton.setOn(False)
        self.tealButton.setOn(False)
        self.greenButton.setOn(False)
        self.redButton.setOn(False)

    def on_maxIntensitiesButton_pressed(self):
        self.violetSlider.setValue(255)
        self.blueSlider.setValue(255)
        self.cyanSlider.setValue(255)
        self.tealSlider.setValue(255)
        self.greenSlider.setValue(255)
        self.redSlider.setValue(255)

    def on_minIntensitiesButton_pressed(self):
        self.violetSlider.setValue(0)
        self.blueSlider.setValue(0)
        self.cyanSlider.setValue(0)
        self.tealSlider.setValue(0)
        self.greenSlider.setValue(0)
        self.redSlider.setValue(0)

    def on_violetButton_toggled(self, val):
        self.violetState = val
        self.osc_send('/1/violetToggle', int(val))
        self.settings['light_engine'].violetState = int(val)
        self.serialChannelEnable(red = self.redState, green=self.greenState,
                                 cyan=self.cyanState, uv=self.violetState,
                                 blue=self.blueState, teal=self.tealState)
            
    def on_blueButton_toggled(self, val):
        self.blueState = val
        self.osc_send('/1/blueToggle', int(val))
        self.settings['light_engine'].blueState = int(val)
        self.serialChannelEnable(red = self.redState, green=self.greenState,
                                 cyan=self.cyanState, uv=self.violetState,
                                 blue=self.blueState, teal=self.tealState)

    def on_cyanButton_toggled(self, val):
        self.cyanState = val
        self.osc_send('/1/cyanToggle', int(val))
        self.settings['light_engine'].cyanState = int(val)
        self.serialChannelEnable(red = self.redState, green=self.greenState,
                                 cyan=self.cyanState, uv=self.violetState,
                                 blue=self.blueState, teal=self.tealState)

    def on_tealButton_toggled(self, val):
        self.tealState = val
        self.osc_send('/1/tealToggle', int(val))
        self.settings['light_engine'].tealState = int(val)
        self.serialChannelEnable(red = self.redState, green=self.greenState,
                                 cyan=self.cyanState, uv=self.violetState,
                                 blue=self.blueState, teal=self.tealState)

    def on_greenButton_toggled(self, val):
        self.greenState = val
        self.osc_send('/1/greenToggle', int(val))
        self.settings['light_engine'].greenState = int(val)
        self.serialChannelEnable(red = self.redState, green=self.greenState,
                                 cyan=self.cyanState, uv=self.violetState,
                                 blue=self.blueState, teal=self.tealState)

    def on_redButton_toggled(self, val):
        self.redState = val
        self.osc_send('/1/redToggle', int(val))
        self.settings['light_engine'].redState = int(val)
        self.serialChannelEnable(red = self.redState, green=self.greenState,
                                 cyan=self.cyanState, uv=self.violetState,
                                 blue=self.blueState, teal=self.tealState)

    def on_greenYellowButton_toggled(self, val):
        self.settings['light_engine'].greenYellowToggle = int(val)
        self.serialGreenYellowFilter(enable = val)

    # ----------------- SLIDERS -------------------

    def on_violetSlider_valueChanged(self, state):
        self.osc_send('/1/violetPercent', '%0.0f%%' % (100 * float(state)/255.0))
        self.osc_send('/1/violet', int(state))
        self.settings['light_engine'].violetVal = int(state)
        self.serialChannelIntensity(3, int(state))

    def on_blueSlider_valueChanged(self, state):
        self.osc_send('/1/bluePercent', '%0.0f%%' % (100 * float(state)/255.0))
        self.osc_send('/1/blue', int(state))
        self.settings['light_engine'].blueVal = int(state)
        self.serialChannelIntensity(4, int(state))

    def on_cyanSlider_valueChanged(self, state):
        self.osc_send('/1/cyanPercent', '%0.0f%%' % (100 * float(state)/255.0))
        self.osc_send('/1/cyan', int(state))
        self.settings['light_engine'].cyanVal = int(state)
        self.serialChannelIntensity(2, int(state))

    def on_tealSlider_valueChanged(self, state):
        self.osc_send('/1/tealPercent', '%0.0f%%' % (100 * float(state)/255.0))
        self.osc_send('/1/teal', int(state))
        self.settings['light_engine'].tealVal = int(state)
        self.serialChannelIntensity(5, int(state))

    def on_greenSlider_valueChanged(self, state):
        self.osc_send('/1/greenPercent', '%0.0f%%' % (100 * float(state)/255.0))
        self.osc_send('/1/green', int(state))
        self.settings['light_engine'].greenVal = int(state)
        self.serialChannelIntensity(1, int(state))

    def on_redSlider_valueChanged(self, state):
        self.osc_send('/1/redPercent', '%0.0f%%' % (100 * float(state)/255.0))
        self.osc_send('/1/red', int(state))
        self.settings['light_engine'].redVal = int(state)
        self.serialChannelIntensity(0, int(state))

# ---------------------------------------------------------------------------------
 
class OscServer(liblo.ServerThread):
    def __init__(self, parent, port):
        liblo.ServerThread.__init__(self, port)
        self.parent = parent

    # ----------------- BUTTONS -------------------

    @liblo.make_method('/1/violetToggle', 'i')
    def violetToggleCallback(self, path, args):
        self.parent.violetButton.setOn(args[0])

    @liblo.make_method('/1/blueToggle', 'i')
    def blueToggleCallback(self, path, args):
        self.parent.blueButton.setOn(args[0])

    @liblo.make_method('/1/cyanToggle', 'i')
    def cyanToggleCallback(self, path, args):
        self.parent.cyanButton.setOn(args[0])

    @liblo.make_method('/1/tealToggle', 'i')
    def tealToggleCallback(self, path, args):
        self.parent.tealButton.setOn(args[0])

    @liblo.make_method('/1/greenToggle', 'i')
    def greenToggleCallback(self, path, args):
        self.parent.greenButton.setOn(args[0])

    @liblo.make_method('/1/redToggle', 'i')
    def redToggleCallback(self, path, args):
        self.parent.redButton.setOn(args[0])

    # ----------------- SLIDERS -------------------

    @liblo.make_method('/1/violet', 'i')
    def violetCallback(self, path, args):
        self.parent.violetSlider.setValue(args[0])

    @liblo.make_method('/1/blue', 'i')
    def blueCallback(self, path, args):
        self.parent.blueSlider.setValue(args[0])

    @liblo.make_method('/1/cyan', 'i')
    def cyanCallback(self, path, args):
        self.parent.cyanSlider.setValue(args[0])

    @liblo.make_method('/1/teal', 'i')
    def tealCallback(self, path, args):
        self.parent.tealSlider.setValue(args[0])

    @liblo.make_method('/1/green', 'i')
    def greenCallback(self, path, args):
        self.parent.greenSlider.setValue(args[0])

    @liblo.make_method('/1/red', 'i')
    def redCallback(self, path, args):
        self.parent.redSlider.setValue(args[0])

    @liblo.make_method(None, None)
    def fallback(self, path, args):
        print "received unknown OSC message '%s'" % path

