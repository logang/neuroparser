"""
Main application for neuroparser
"""
# Disable opengl error logging
import OpenGL
OpenGL.ERROR_CHECKING = False

import sys
import os
import os.path

# Initial import of the settings object.  This object is used
# throughout the codebase to store application state and preferences.
#
# If you are looking for the location of the settings files, please
# refer to the "Platform Specific Notes" at
# http://doc.qt.nokia.com/latest/qsettings.html
#
# Note: It's important that we import the settings before anything else Qt
# related, because the settings sets the SIP API to version 2.
# Hopefully this won't matter in the future when the transition to SIP
# 2 is complete.
#
from settings import Settings
settings = Settings()

# Set the system path.  This helps scope.py to find the wrapped C++
# modules, and allow eggs to be dropped into application folder, as
# well as script overrides, etc.
sys.path = [os.getcwd(),
            os.path.join(os.getcwd(), '..', 'optimization'),
            os.path.join(os.getcwd(), '..', 'optimization', 'cwpath')] + sys.path

# Set up some basic logging (for experiments & program debugging).  
#from log import xenon_log, xenon_warning, xenon_error

# Create the application
from PyQt4 import QtCore, QtGui
app = QtGui.QApplication(sys.argv)

# Set defaults
if not settings['output'].contains('default_path'):
    settings['output'].default_path = os.getcwd()
if not settings['input'].contains('default_path'):
    settings['input'].default_path = os.getcwd()

# load resource location
cwd = os.getcwd()
if not sys.argv[0]:
    resource_path = cwd
else:
    resource_path = os.path.dirname(os.path.abspath(sys.argv[0]))
settings['app'].resource_path = resource_path

# Show the splash screen
#splash_path = os.path.join(settings['app'].resource_path, 'splash.png')
#splash = QtGui.QSplashScreen(QtGui.QPixmap(splash_path))
#splash.show()

# set up my application
QtCore.QCoreApplication.setOrganizationName('Stanford University')
QtCore.QCoreApplication.setOrganizationDomain('stanford.edu')
QtCore.QCoreApplication.setApplicationName('Neuroparser')

# ----------       SUPPORTED CAMERAS  -------------
#camera_labels_to_initparams = {"demo_cam": ("demo_cam", "DemoCamera", "DCam"),
#                               "andor"   : ("andor", "Andor3Camera", "Neo"),
#                               "qcam"    : ("qcam", "QCam", "QCamera"),
#                               "sapera"  : ("sapera", "SaperaCamera", "SaperaCam")}
#camera_labels = camera_labels_to_initparams.keys()
#string_camera_list = ", ".join(camera_labels)

# ---------- PARSE COMMAND LINE OPTIONS -------------

from optparse import OptionParser

# Refresh default application settings. 
#settings['app'].refreshWithDefault('selected_camera', 'demo_cam')
#settings['app'].refreshWithDefault('circular_buffer_size', 1024)

parser = OptionParser()
parser.add_option('', "--debug",
                  action="store_true", dest="debug", default=True,
                  help="Turn on debugging output.")

# parser.add_option("-b", "--circular-buffer-size", dest="circular_buffer_size", action='store', type='int',
#                   default=settings['app'].circular_buffer_size,
#                   help="Specify the size of the circular buffer (in MB)")
# parser.add_option("-c", "--camera",
#                   action="store", type='string', dest="selected_camera", default=settings['app'].selected_camera,
#                   help="Select which camera driver to use.  Your options are: [" + string_camera_list + "]")
(options, args) = parser.parse_args()

# Parse circular buffer size argument
#if options.circular_buffer_size:
#    assert (options.circular_buffer_size > 128), "Circular buffer size must be >= 128MB"
#    settings['app'].circular_buffer_size = options.circular_buffer_size

# Parse camera argument
#if options.selected_camera:
#    assert (options.selected_camera in camera_labels), "Camera not found in allowed camera list:" + string_camera_list
#    settings['app'].selected_camera = options.selected_camera

# TODO: Possibly make these dictionaries populated by some 
# metadata file in the camera driver directory?
#camera_allowed_os = {"demo_cam": ("darwin", "win32", "linux2"),
#                     "andor"   : ("win32", "linux2"),
#                     "qcam"    : ("win32",),
#                     "sapera"  : ("win32",)}

#assert os.sys.platform in camera_allowed_os[settings['app'].selected_camera], "The camera (%s) is not supported by your operating system: %s" % (settings['app'].selected_camera, os.sys.platform)

# Create the framegrabber object, then load & initialize the camera device.
#from py_xenon_framegrabber import FrameGrabber, DiskFrameStore
#frameGrabber = FrameGrabber()
#try:
#    frameGrabber.loadDevice(*camera_labels_to_initparams[settings['app'].selected_camera])
#    frameGrabber.initializeDevice(settings['app'].selected_camera)
#except Exception as inst:
#    print "\n\nAn error occurred while trying to initialize the \""
#    print settings['app'].selected_camera + "\" camera:\n" + inst + '\nExiting.'
#    sys.exit(0)

# --------------- FINAL APPLICATION SETUP ----------------------

#splash.showMessage('Loading...', QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)

# The disk framestore object creates a thread that saves images to
# disk as they become available in the MM circular buffer.
#diskFrameStore = DiskFrameStore()

# The plugin manager loads and manages rending and event handling for
# stimulation plugins.
#from plugins.PluginManager import PluginManager
#pluginManager = PluginManager()

# Create the mainwindow
import mainwindow
mainWindow = mainwindow.MainWindow() #frameGrabber, diskFrameStore, pluginManager, settings['app'].selected_camera)
mainWindow.show()
#splash.finish(mainWindow)

# Set the circular buffer size
#frameGrabber.setCircularBufferMemoryFootprint(settings['app'].circular_buffer_size) # MB
#print '\t--> Circular buffer capacity: ', frameGrabber.getBufferTotalCapacity(), ' images.'

# run the application, and then exit when the runloop terminates.
result = app.exec_()

# Shut down micromanager gracefully.  This is important for some
# cameras (especially QCam), which can easily be be left in an
# inconsistent state.
#frameGrabber.stopSequenceAcquisition()

# Delete the mainwindow.  This seems to cause memory to be cleaned up
# nicely.  Otherwise we see strange errors as python's garbage
# collection cleans up objects in an order that will sometimes lead to
# bad OpenGL contexts getting called or other strange problems.
del mainWindow

# Shut down the disk framestore thread gracefully.
#diskFrameStore.shutdown()

# Finally, we shut down the framegrabber which should safely terminate
# micro-manager, among other things.
#frameGrabber.shutdown()

