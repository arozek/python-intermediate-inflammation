"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data (each row contains measurements for a single patient across all days).
    :returns: An array of mean values of measurements for each day.
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data (each row contains measurements for a single patient across all days).
    :returns: An array of maximum values of measurements for each day.
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.
    :param data: A 2D data array with inflammation data (each row contains measurements for a single patient across all days).
    :returns: An array of minimum values of measurements for each day.
    """
    return np.min(data, axis=0)

def patient_normalise(data):
    """Normalise patient inflammation data from a 2D inflammation data array.
    :param data: A 2D data array with inflammation data (each row contains measurements for a single patient across all days).
    :returns: An 2D array of patient inflammation data normalised to the maximum value
    """
    if not isinstance(data,np.ndarray):
        raise TypeError('Wrong data type')
    if len(data.shape) != 2:
        raise ValueError('Inflammation array should be 2D')
    if np.any(data<0):
        raise ValueError('Inflammation values should not be negative')

    max_data = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    return normalised

def attach_names(data,names):
    """Create data structure containing patient records"""
    assert len(data)==len(names)
    output=[]
    for data_row,name in zip(data,names):
        output.append({'name': name,
                       'data': data_row})
    return output

class Observation:
    def __init__(self, day, value):
        self.day=day
        self.value=value

    def __str__(self):
        return str(self.value)

class Person:
    """A random person that just has a name"""
    def __init__(self,name):
        if not isinstance(name, str):
            raise TypeError('Name must be a string')
            self.name=None
        else:
            self.name=name
    def __str__(self):
        return self.name

class Patient(Person):
    """A patient in an inflammation study. Who is also a person"""
    def __init__(self, name, observations=None):
        super().__init__(name)
        if observations is None:
            self.observations = []
        else:
            self.observations=observations

    def add_observation(self,value,day=None):
        if day is None:
            try:
                day = self.observations[-1].day + 1
            except IndexError:
                day = 0

            new_observation=Observation(day,value)

            self.observations.append(new_observation)
            return(new_observation)

    @property
    def last_observation(self):
        return self.observations[-1]

class Doctor(Person):
    def __init__(self,name):
        super().__init__(name)
        self.patients=[]

    def add_patient(self, new_patient):

        if isinstance(new_patient, str):
            self.patients.append(Patient(new_patient))
        elif isinstance(new_patient, Patient):
            self.patients.append(new_patient)
        else:
            raise TypeError("A patient must be a Patient or patient's name")

    def patient_ID(self,patient_name):
        for i in range(len(self.patients)):
            if self.patients[i].name ==patient_name:
                return i
        return None

#    def add_patient(self, patient,patient_id=None)

