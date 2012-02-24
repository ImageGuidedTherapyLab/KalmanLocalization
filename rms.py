import vtk
# echo vtk version info
print "using vtk version", vtk.vtkVersion.GetVTKVersion()
import vtk.util.numpy_support as vtkNumPy 
import numpy
import os
import scipy.io as scipyio

# return real and imaginary dat from from magn phase data 
def GetNumPyData(filename):
  print filename
  vtiReader = vtk.vtkXMLImageDataReader()
  vtiReader.SetFileName(filename)
  vtiReader.Update()
  vtiData = vtk.vtkImageCast()
  vtiData.SetOutputScalarTypeToDouble()
  vtiData.SetInput( vtiReader.GetOutput() )
  vtiData.Update( )
  vti_image = vtiData.GetOutput().GetPointData() 
  vti_array = vtkNumPy.vtk_to_numpy(vti_image.GetArray(0)) 
  return vti_array 

rootdir = "%s/kalmanNoROI/" % os.getcwd()
ntime=128
deltat = 5.0

rmsValues = []
normrmsValues = []
for timeID in range(0,ntime):
  timeValue = [timeID*deltat]
  normValue = [timeID*deltat]
  for dataID in range(1,18+1):
    mrtifile   = "%s/mrtiROI5.%04d.%04d.vti"  %(rootdir,dataID,timeID)
    femfile    = "%s/updatefemROI5.%04d.%04d.vti"  %(rootdir,dataID,timeID)
    stddevfile = "%s/stddevROI5.%04d.%04d.vti"%(rootdir,1,timeID)
    try: 
      if( not os.path.isfile(mrtifile  )):
        raise
      if( not os.path.isfile(femfile)):
        raise
      if( not os.path.isfile(stddevfile)):
        raise
      # get data
      mrti  =GetNumPyData(mrtifile  )
      fem   =GetNumPyData(femfile   )
      stddev=GetNumPyData(stddevfile)
      error   = mrti-fem
      sqError = error*error
      #normalize
      normerror  =     error * numpy.reciprocal( stddev)
      sqNormerror= normerror * normerror  
      timeValue = timeValue + [numpy.sqrt(    sqError.sum()/len(sqError))] 
      normValue = normValue + [numpy.sqrt(sqNormerror.sum()/len(sqNormerror)) ]
    except: 
      timeValue = timeValue + [ 0.0]
      normValue = normValue + [ 0.0]
  rmsValues.append( timeValue )
  normrmsValues.append( normValue )
numpy.savetxt("rms.dat" , rmsValues, fmt="%.16e")
numpy.savetxt("normrms.dat" , normrmsValues, fmt="%.16e")
