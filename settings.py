
# First we set the SIP API to force PyQt to use the more modern
# QStrings and QVariants that are transparently converted into python
# strings.
#
# Note: in order for this change to take effect, you need to import
# settings.py *before* importing any other QT headers into your
# program.  It's easiest to put this import statement at the very top
# of your main program file.
import sip
sip.setapi('QString', 2)
sip.setapi('QVariant', 2)
from PyQt4 import QtCore

class Settings(object):
    """
    The settings class manages state that persists between program
    instances.

    The settings class uses separate namespaces to help keep settings
    organized.  The syntax for using settings (and namespaces) is:

      from settings import Settings
      settings = settings.Settings()
      settings['<namespace>'].<attribute>  # Access attribute in namespace
      settings.<attribute>                 # assumes default namespace of 'app'

    This settings class is fully re-entrant, and settings
    can even be shared across multiple running instances of the
    program.  For more information, refer to the section on "Accessing Settings
    from Multiple Threads or Processes Simultaneously" at
    http://www.opendocs.net/pyqt/pyqt4/html/qsettings.html
    """
    def __init__(self):
        self._qt_settings = QtCore.QSettings(QtCore.QSettings.IniFormat,
                                             QtCore.QSettings.UserScope,
                                             'Stanford University', 'uScope')
        self._namespaces = {}
        self.__initialised = True  # for __setattr__

    def __getitem__(self, namespace):
        if not self._namespaces.has_key(namespace):
            self._namespaces[namespace] = SettingsNamespace(self._qt_settings, namespace)
        return self._namespaces[namespace]

    def __getattr__(self, key):
        # any normal attributes are handled normally
        if self.__dict__.has_key(key):
            return dict.__getattr__(self, key)
        else:
            return self["app"].__getattr__(key)
        
    def __setattr__(self, key, value):
        # this test allows attributes to be set in the __init__ method
        if not self.__dict__.has_key('_Settings__initialised'):
            return dict.__setattr__(self, key, value)

        # any normal attributes are handled normally
        elif self.__dict__.has_key(key):
            dict.__setattr__(self, key, value)

        # And the remaining attributes are passed along to the general namespace class
        else:
            return self["app"].__setattr__(key, value)

    def setValue(self, key, value):
        raise NotImplementedError("You cannot use the setValue() method with the settings object.")

    def value(self, key):
        raise NotImplementedError("You cannot use the value() method with the settings object.")

    def sync(self):
        self._qt_settings.sync()

# --------------- SettingsNamespace ------------------
#
# You only use this class indirectly by calling
# settings['<namespace>'].<attribute>.

class SettingsNamespace(object):

    def __init__(self, qt_settings, namespace):
        self._qt_settings = qt_settings
        self.namespace = namespace
        self.__initialised = True  # for __setattr__

    # To make the code cleaner, we will expose all of the Qt settings
    # as attributes of the settings class. 
    def __getattr__(self, key):
        fully_qualified_key = self.namespace + '/' + key
        fully_qualified_keytype = self.namespace + '/' + "_" + key + "_type"
        if (not self._qt_settings.contains(fully_qualified_key)):
            raise AttributeError("Settings object does not contain the key \"" + str(fully_qualified_key) + "\"")

        # QSettings does not save the type of bool, int, and float
        # arguments, so we must decode their type manually here.  This
        # is a pain, but it works.
        value_type = self._qt_settings.value(fully_qualified_keytype)
        if (value_type == 'bool'):
            if (self._qt_settings.value(fully_qualified_key) == 'True' or
                self._qt_settings.value(fully_qualified_key) == 'true' or
                self._qt_settings.value(fully_qualified_key) == True):
                return True
            else:
                return False
        elif (value_type == 'int'):
            return int(self._qt_settings.value(fully_qualified_key))
        elif (value_type == 'float'):
            return float(self._qt_settings.value(fully_qualified_key))
        else:
            return self._qt_settings.value(fully_qualified_key)

    def __setattr__(self, key, value):
        # this test allows attributes to be set in the __init__ method
        if not self.__dict__.has_key('_SettingsNamespace__initialised'):
            return dict.__setattr__(self, key, value)

        # QSettings does not save the type of bool, int, and float
        # arguments, so we must encode their type manually here.  This
        # is a pain, but it works.
        value_type = str(type(value))
        type_key = self.namespace + '/' + "_"+key+"_type"
        if (value_type == "<type \'bool\'>"):
            self._qt_settings.setValue(type_key, 'bool')
        elif (value_type == "<type \'int\'>"):
            self._qt_settings.setValue(type_key, 'int')
        elif (value_type == "<type \'float\'>"):
            self._qt_settings.setValue(type_key, 'float')

        # Set the value of the key in the data file.c
        return self._qt_settings.setValue(self.namespace + '/' + key, value)

    def contains(self, key):
        fully_qualified_key = self.namespace + '/' + key
        return self._qt_settings.contains(fully_qualified_key)

    def valueWithDefault(self, key, default_value):
        fully_qualified_key = self.namespace + '/' + key
        if (not self._qt_settings.contains(fully_qualified_key)):
            return default_value
        else:
            return self.__getattr__(key)

    def refreshWithDefault(self, key, default_value):
        fully_qualified_key = self.namespace + '/' + key
        if (not self._qt_settings.contains(fully_qualified_key)):
            self.__setattr__(key, default_value)
            return default_value
        else:
            # Try to access the setting.  If it fails for some reason, return the default value.
            try:
                return self.__getattr__(key)
            except TypeError:
                return default_value

    def setValue(self, key, value):
        self.__setattr__(key, value)

    def value(self, key):
        return self.__getattr__(key)
