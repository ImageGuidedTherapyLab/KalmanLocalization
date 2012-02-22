# Read DAKOTA parameters file (aprepro or standard format) and call a
# Python module for fem analysis.

# DAKOTA will execute this script as
#   deltapModeling.py params.in results.out
# so sys.argv[1] will be the parameters file and
#    sys.argv[2] will be the results file to return to DAKOTA

# necessary python modules
import sys
import re
import os
import math

# write a numpy data to disk in vtk format
def ConvertNumpyVTKImage(NumpyImageData,ArrayName,ImageDim,ImageSpacing,ImageOrigin):
  # Create initial image
  # imports raw data and stores it.
  import vtk
  dataImporter = vtk.vtkImageImport()
  # array is converted to a string of chars and imported.
  data_string = NumpyImageData.tostring()
  dataImporter.CopyImportVoidPointer(data_string, len(data_string))
  # The type of the newly imported data is set to unsigned char (uint8)
  dataImporter.SetDataScalarTypeToDouble()
  # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
  # must be told this is the case.
  dataImporter.SetNumberOfScalarComponents(1)
  # The following two functions describe how the data is stored and the dimensions of the array it is stored in. For this
  # simple case, all axes are of length 75 and begins with the first element. For other data, this is probably not the case.
  # I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
  # VTK complains if not both are used.
  dataImporter.SetDataExtent( 0, ImageDim[0]-1, 0, ImageDim[1]-1, 0, ImageDim[2]-1)
  dataImporter.SetWholeExtent(0, ImageDim[0]-1, 0, ImageDim[1]-1, 0, ImageDim[2]-1)
  dataImporter.SetDataSpacing( ImageSpacing )
  dataImporter.SetDataOrigin(  ImageOrigin)
  dataImporter.SetScalarArrayName(ArrayName)
  dataImporter.Update()
  return dataImporter.GetOutput()
  
def KalmanFilterMRTI(**kwargs):
  """
  kalman filtered MRTI 
  """
  # import needed modules
  import petsc4py, numpy
  PetscOptions =  sys.argv
  PetscOptions.append("-ksp_monitor")
  PetscOptions.append("-ksp_rtol")
  PetscOptions.append("1.0e-15")
  PetscOptions.append("-solver")
  #PetscOptions.append("petsc")
  PetscOptions.append("superlu_dist")
  PetscOptions.append("-log_summary")
  PetscOptions.append("-override_max")
  PetscOptions.append("-modelcov")
  PetscOptions.append( kwargs['cv']['modelcov'] )
  #PetscOptions.append("-verify_inverse")
  #PetscOptions.append("-write_system_matrix")
  #PetscOptions.append("-ksp_inverse_pc")
  #PetscOptions.append("ilu")
  #PetscOptions.append("lu")
  # ROI info
  #PetscOptions.append("-nx")
  #PetscOptions.append("-35")
  #PetscOptions.append("-ny")
  #PetscOptions.append("-35")
  #PetscOptions.append("-ix")
  #PetscOptions.append("-125")
  #PetscOptions.append("-iy")
  #PetscOptions.append("-115")
  #PetscOptions.append("-help")
  petsc4py.init(PetscOptions)
  
  from petsc4py import PETSc
  
  # create stages for logging
  PredictionStage = PETSc.Log.Stage("Prediction")
  CorrectionStage = PETSc.Log.Stage("Correction")
  
  # break processors into separate communicators
  petscRank = PETSc.COMM_WORLD.getRank()
  petscSize = PETSc.COMM_WORLD.Get_size()
  sys.stdout.write("petsc rank %d petsc nproc %d\n" % (petscRank, petscSize))
  
  # set shell context
  # TODO import vtk should be called after femLibrary ???? 
  # FIXME WHY IS THIS????
  import femLibrary
  # initialize libMesh data structures
  libMeshInit = femLibrary.PyLibMeshInit(PetscOptions,PETSc.COMM_WORLD) 
    
  # store control variables
  getpot = femLibrary.PylibMeshGetPot(PetscOptions) 
  #getpot.SetIniValue( "thermal_conductivity/k_0_probe","1.0e8") 
  #getpot.SetIniValue( "thermal_conductivity/k_0_tumor","1.0e8") 
  getpot.SetIniValue( "perfusion/w_0_healthy", "9.0"  ) 
  # tumor = applicator
  getpot.SetIniValue( "perfusion/w_0_tumor", "0.0"  ) 
  getpot.SetIniValue( "optical/mu_a_healthy",  "5.0e2") 
  getpot.SetIniValue( "optical/mu_s_healthy","140.0e2") 
  getpot.SetIniValue( "optical/anfact"      ,  "0.88" ) 
  # from Duck table 2.15
  getpot.SetIniValue( "material/specific_heat","3840.0" ) 
  # set ambient temperature 
  u_init = 37.0
  probe_init = 21.0
  probe_init = u_init 
  getpot.SetIniValue( "initial_condition/u_init","%f" % u_init ) 
  getpot.SetIniValue( "initial_condition/probe_init","%f" % probe_init ) 
  getpot.SetIniValue( "bc/u_dirichletid","1" ) #apply dirichlet data on applicator domain
  # set probe domain for heating
  getpot.SetIniValue( "probe/domain" , "2" ) 
  
  # root directory of data
  # FIXME Josh update with your data directory
  workDir = "/share/work/fuentes/data/biotex/090318_751642_treat/"
  workDir = "/data/fuentes/biotex/090318_751642_treat/"
  workDir = "/work/00131/jyung/data/biotex/KalmanLocalization/"
  dataRoot = "%s/Processed/s1%02d%03d" % ( workDir, kwargs['cv']['uniform'] , kwargs['cv']['roi'] )
  tmapRoot = "%s/Processed/s1%02d%03d" % ( workDir,          0              ,          0          )

  # load vtk modules to read imaging
  import vtk 
  import vtk.util.numpy_support as vtkNumPy 
  # read imaging data geometry that will be used to project FEM data onto
  #vtkReader = vtk.vtkXMLImageDataReader() 
  vtkReader = vtk.vtkDataSetReader() 
  vtkReader.SetFileName('%s/tmap.0000.vtk' % tmapRoot )
  vtkReader.Update()
  templateImage = vtkReader.GetOutput()
  dimensions = templateImage.GetDimensions()
  # dimensions should already be in meters
  spacing = [ dx * 1.000 for dx in templateImage.GetSpacing() ] 
  origin  = [ x0 * 1.000 for x0 in templateImage.GetOrigin() ] 
  origin  = [-0.095223505, -0.058349645000000006, 0.049946399000000002]
  print spacing, origin, dimensions
  femImaging = femLibrary.PytttkImaging(getpot, dimensions ,origin,spacing) 

  image_roi = [[130,155],[123,148],[0,0]] 
  size_roi  = [ image_roi[0][1]-image_roi[0][0]+1,
                image_roi[1][1]-image_roi[1][0]+1,
                image_roi[2][1]-image_roi[2][0]+1]
  roiOrigin =  [ origin[0] + image_roi[0][0] * spacing[0],
                 origin[1] + image_roi[1][0] * spacing[1],
                 origin[2] + image_roi[2][0] * spacing[2]]

  # create ROI template
  roi_array = numpy.zeros(dimensions,dtype=numpy.double,order='C')
  #roi_array[118:153,125:160] = 1.e4 
  # TODO row/column major ordering never seems to work...
  # FIXME notice the indices are reversed to account to the transpose 
  roi_array[image_roi[1][0]:image_roi[1][1]+1,
            image_roi[0][0]:image_roi[0][1]+1] = 1.e4 
  roi_vec = PETSc.Vec().createWithArray( roi_array, comm=PETSc.COMM_SELF)
  
  # initialize FEM Mesh
  femMesh = femLibrary.PylibMeshMesh()
  RotationMatrix = [[1.,0.,0.],
                    [0.,1.,0.],
                    [0.,0.,1.]]
  Translation =     [0.,0.,0.]
  #Setup Affine Transformation for registration
  AffineTransform = vtk.vtkTransform()
  #AffineTransform.Translate( [0.050,0.080, origin[2]] )
  AffineTransform.Translate( [0.051,0.080, 0.0509] )
  AffineTransform.RotateZ( 29.0 )
  AffineTransform.RotateY( 86.0 )
  #AffineTransform.RotateY( 90.0 )
  AffineTransform.RotateX(  0.0 )
  AffineTransform.Scale([1.,1.,1.])
  matrix = AffineTransform.GetConcatenatedTransform(0).GetMatrix()
  RotationMatrix = [[matrix.GetElement(0,0),matrix.GetElement(0,1),matrix.GetElement(0,2)],
                    [matrix.GetElement(1,0),matrix.GetElement(1,1),matrix.GetElement(1,2)],
                    [matrix.GetElement(2,0),matrix.GetElement(2,1),matrix.GetElement(2,2)]]
  Translation =     [matrix.GetElement(0,3),matrix.GetElement(1,3),matrix.GetElement(2,3)] 
  # set intial mesh
  #femMesh.SetupUnStructuredGrid( "%s/meshTemplate1.e" % workDir ,0,RotationMatrix, Translation  ) 
  femMesh.SetupUnStructuredGrid( "%s/meshTemplate2.NoAppBoundary.e" % workDir ,0,RotationMatrix, Translation  ) 
  #femMesh.SetupUnStructuredGrid( "%s/meshTemplate2.WithAppBoundary.e" % workDir ,0,RotationMatrix, Translation  ) 
  #femMesh.SetupStructuredGrid( (60,60,2), 
  #                             [.0216,.0825],[.0432,.104],[-.001,.001],[2,2,2,2,3,2]) 
  #femMesh.ReadFile("magnitudeROI.e")
  
  # add the data structures for the Background System Solve
  # set deltat, number of time steps, power profile, and add system
  acquisitionTime = 5.00
  deltat = acquisitionTime 
  ntime  = 128 
  eqnSystems =  femLibrary.PylibMeshEquationSystems(femMesh,getpot)
  getpot.SetIniPower(1, [ [17,27,39,69,ntime],[0.0,4.05,0.0,10.05,0.0] ])
  
  # instantiate Kalman class
  kalmanFilter = None
  if ( kwargs['cv']['algebra'] == 0  ):
    kalmanFilter = femLibrary.PytttkDenseKalmanFilterMRTI(eqnSystems, deltat) 
  elif ( kwargs['cv']['algebra'] == 1  ):
    kalmanFilter = femLibrary.PytttkUncorrelatedKalmanFilterMRTI(eqnSystems, deltat) 
  elif ( kwargs['cv']['algebra'] == 2  ):
    kalmanFilter = femLibrary.PytttkSparseKalmanFilterMRTI(eqnSystems, deltat) 
  else:
    raise RuntimeError("\n\n unknown linear algebra... ")
  kalmanFilter.systems["StateSystem"].AddStorageVectors(ntime)
  
  # add systems to plot covariance 
  CovCol = [0,1,4,5,8,10,100,101,104,150,151,154,200,201,204,250,251,254,300,301,304,350,351,354,400,401,404,450,451,454,504,505,510,900,901,904,905,908,910,1000,1001,1004,1050,1051,1054,1100,1101,1104,1150,1151,1154,1200,1201,1204,1250,1251,1254,1300,1301,1304,1350,1351,1354,1400,1401,1404,1450,1451,1454,1500,1501,1504,1550,1551,1554,1600,1601,1604,1650,1651,1654,1700,1701,1704,1750,1751,1754,1802,1803,1805,1807,1832,1835,1836,1839,1908,1911,1946,1949,1984,1987,2022,2025,2060,2063,2098,2101,2136,2139,2174,2177,2216,2219,2516,2519,2520,2523,2592,2595,2630,2633,2668,2671,2706,2709,2744,2747,2782,2785,2820,2823,2858,2861,2896,2899,2934,2937,2972,2975,3010,3013,3048,3051,3086,3089,3124,3127,3162,3165,3201,3203,3217,3219,3221,3223,3303,3305,3346,3348,3389,3391,3432,3434,3475,3477,3518]
  CovCol = [3520,3561,3563,3604,3606,3649,3653,3991,3993,3995,3997,4077,4079,4120,4122,4163,4165,4206,4208,4248,4250,4291,4293,4334,4336,4377,4379,4420,4422,4463,4465,4506,4508,4549,4551,4592,4594,4635,4637,4678,4680,4721,4723,4766,4767,4780,4782,4842,4873,4904,4935,4966,4997,5028,5059,5092,5338,5340,5400,5431,5462,5493,5524,5555,5586,5617,5648,5679,5710,5741,5772,5803,5834,5865,5897,5904,5905,5906,5907,5908,5909,5910,5911,5912,5913,5914,5915,5916,5917,5918,5919,5920,5921,5922,5923,5924,5925,5926,5927,5928,5929,5930,5931,5932,5933,5934,5935,5936,5937,5938,5939,5940,5941,5942,5943,5944,5945,5946,5947,5948,5949,5950,5951,5952,5953,5954,5955,5956,5957,5958,5959,5960,5961,5962,5963,5964,5965,5966,5967,5968,5969,5970,5971,5972,5973,5974,5975,5976,5977,5978,5979,5980,5981,5982,5983,5984,5985,5986,5987,5988,5989,5990,5991,5992,5993,5994,5995,5996,5997,5998,5999,6000]
  CovCol = [6001,6002,6003,6004,6005,6006,6007,6008,6009,6010,6011,6012,6013,6014,6015,6016,6017,6018,6019,6020,6021,6022,6023,6024,6025,6026,6027,6028,6029,6030,6031,6032,6033,6034,6035,6036,6037,6038,6039,6040,6041,6042,6043,6044,6045,6046,6047,6048,6049,6050,6051,6052,6053,6054,6055,6056,6057,6058,6059,6060,6061,6062,6063,6064,6065,6066,6067,6068,6069,6070,6071,6072,6073,6074,6075,6076,6077,6078,6079,6080,6081,6082,6083,6084,6085,6086,6087,6088,6089,6090,6091,6092,6093,6094,6095,6096,6097,6098,6099,6100,6101,6102,6103,6104,6105,6106,6107,6108,6109,6110,6111,6112,6113,6114,6115,6116,6117,6118,6119,6120,6121,6122,6123,6124,6125,6126,6127,6128,6129,6130,6131,6132,6133,6134,6135,6136,6137,6138,6139,6140,6141,6142,6143,6144,6145,6146,6147,6148,6149,6150,6151,6152,6153,6154,6155]
  CovCol = [33,187,3469]
  CovCol = CovCol + [40,69,111,222,555,654,807,911,987,5454]

  for column in CovCol: 
   tmpCovSystem =  eqnSystems.AddExplicitSystem( "Cov%04d" % column ) 
   tmpCovSystem.AddFirstLagrangeVariable( "col%04d" % column ) 
  
  preUpdateSystem =  eqnSystems.AddExplicitSystem( "preUpdate" ) 
  preUpdateSystem.AddFirstLagrangeVariable( "predict" ) 
  preUpdateSystem.AddFirstLagrangeVariable( "damage" ) 
  preUpdateSystem.AddFirstLagrangeVariable( "damderiv" ) 

  roiSystem = eqnSystems.AddExplicitSystem( "ROISystem" ) 
  roiSystem.AddFirstLagrangeVariable( "roi" ) 

  # initialize libMesh data structures
  eqnSystems.init( ) 
  
  # print info
  eqnSystems.PrintSelf() 
  
  # project ROI 
  femImaging.ProjectImagingToFEMMesh("ROISystem",0.0,roi_vec,eqnSystems)  

  # show interpolations range
  roi_vec.set(1.e4)
  femImaging.ProjectImagingToFEMMesh("MRTIMean",0.0,roi_vec,eqnSystems)  

  # choose system to create measurement from
  #numMeasurement = kalmanFilter.CreateMeasurementMapFromImaging("MRTIMean", 8 )
  #numMeasurement = kalmanFilter.CreateMeasurementMapFromImaging("ROISystem", 8 )

  # create identity map if needed 
  #numMeasurement = kalmanFilter.CreateIdentityMeasurementMap( 9 )

  # initialize petsc data structures
  numMeasurement = size_roi[0]* size_roi[1]* size_roi[2] 
  kalmanFilter.Setup( numMeasurement )
   
  MeasurementMapNodeSet = 9
  MeasurementMapNodeSet = 7
  MeasurementMapNodeSet = 8
  #kalmanFilter.CreateROINodeSetMeasurementMap( MeasurementMapNodeSet )
  MeasurementMapNodeSet = 5 # use node set ID as the averaging number
  kalmanFilter.CreateROIAverageMeasurementMap(image_roi, MeasurementMapNodeSet , 
                                              [origin[0], origin[1], 0.0469], spacing )

  # get covariance diagonal and and covariance entries for plotting
  for column in CovCol:
    kalmanFilter.ExtractCoVarianceForPlotting("Cov%04d" % column,column)
  kalmanFilter.ExtractVarianceForPlotting("StateStdDev")

  # set output file
  MeshOutputFile = "fem_data%s%d.%04d.e" % (kalmanFilter.LinAlgebra,
                                            MeasurementMapNodeSet , kwargs['fileID'] )

  # write IC
  exodusII_IO = femLibrary.PylibMeshExodusII_IO(femMesh)
  exodusII_IO.WriteTimeStep(MeshOutputFile,eqnSystems, 1, 0.0 )  
  
  # NOTE This is after IC write
  # error check that can recover a constant field
  roiVec = roiSystem.GetSolutionVector()
  roiVec.set(1000.0)
  imagetest = kalmanFilter.ProjectMeasurementMatrix(roiVec)

  # show transpose measurement matrix map
  kalmanFilter.MeasurementMatrixTranspose("ROISystem",1.e3)

  # loop over time steps and solve
  #for timeID in range(35,70):
  for timeID in range(1,ntime):
     print "time step = " ,timeID
     eqnSystems.UpdatePetscFEMSystemTimeStep("StateSystem",timeID ) 
     PredictionStage.push() # log prediction stage
     # use model to predict the state and covariance
     kalmanFilter.StatePredict(timeID)
     kalmanFilter.CovariancePredict()
     PredictionStage.pop() # log prediction stage

     # copy from state system
     preUpdateSystem.CopySolutionVector( kalmanFilter.systems["StateSystem"] )
  
     CorrectionStage.push() # log correction stage
  
     # read MRTI Data
     vtkTmapReader = vtk.vtkDataSetReader() 
     vtkTmapReader.SetFileName('%s/tmap.%04d.vtk' % (tmapRoot,timeID) )
     vtkTmapReader.Update() 
     tmap_cells = vtkTmapReader.GetOutput().GetPointData()
     tmap_array = u_init + vtkNumPy.vtk_to_numpy( tmap_cells.GetArray('scalars') ) 
     tmap_vec = PETSc.Vec().createWithArray( tmap_array, comm=PETSc.COMM_SELF)
     femImaging.ProjectImagingToFEMMesh("MRTIMean",u_init,tmap_vec,eqnSystems)  
  
     # read in SNR base uncertainty measurement
     measurementCov = 2.0 * 2.0 ; 
     vtkSTDReader = vtk.vtkDataSetReader() 
     vtkSTDReader.SetFileName('%s/snruncert.%04d.vtk' % (dataRoot,timeID) )
     vtkSTDReader.Update() 
     std_cells = vtkSTDReader.GetOutput().GetPointData() 
     snr_array = vtkNumPy.vtk_to_numpy(std_cells.GetArray('scalars')) 
     snr_vec = PETSc.Vec().createWithArray( snr_array, comm=PETSc.COMM_SELF )
     femImaging.ProjectImagingToFEMMesh("MRTIStdDev",measurementCov,snr_vec,eqnSystems)  
  
     # extract data to kalman data structures
     #mrtiSoln = kalmanFilter.systems["MRTIMean"].GetSolutionVector( )
     #mrtiROI = kalmanFilter.ProjectMeasurementMatrix("MRTIMean")
     mrtiROI = PETSc.Vec().createWithArray( 
                    tmap_array.reshape(dimensions)[image_roi[1][0]:image_roi[1][1]+1,
                                                   image_roi[0][0]:image_roi[0][1]+1],comm=PETSc.COMM_SELF)
     kalmanFilter.ExtractMeasurementData(mrtiROI,"MRTIStdDev")
     #kalmanFilter.systems["MRTIMean"].ApplyDirichletData()
     #kalmanFilter.systems["MRTIStdDev"].ApplyDirichletData()
  
     # save prefem to disk for dbg
     PreFEMsoln = eqnSystems.GetPetscFEMSystemSolnSubVector( "StateSystem", 0)
     predictFEM = kalmanFilter.ProjectMeasurementMatrix(PreFEMsoln)

     vtkPreFEMImage = ConvertNumpyVTKImage(predictFEM[...],"prefem",size_roi,spacing,roiOrigin)
     vtkPreFEMWriter = vtk.vtkXMLImageDataWriter()
     vtkPreFEMWriter.SetFileName( "prefemROI%d.%04d.%04d.vti" % (MeasurementMapNodeSet, kwargs['fileID'],timeID) )
     vtkPreFEMWriter.SetInput( vtkPreFEMImage )
     vtkPreFEMWriter.Update()

     # save mrti to disk for compare
     vtkMRTImage = ConvertNumpyVTKImage(mrtiROI[...],"mrti",size_roi,spacing,roiOrigin)
     vtkMRTIWriter = vtk.vtkXMLImageDataWriter()
     vtkMRTIWriter.SetFileName( "mrtiROI%d.%04d.%04d.vti" % (MeasurementMapNodeSet, kwargs['fileID'],timeID) )
     vtkMRTIWriter.SetInput( vtkMRTImage )
     vtkMRTIWriter.Update()

     # save stddev to disk for rms
     stddevROI = PETSc.Vec().createWithArray( 
                    snr_array.reshape(dimensions)[image_roi[1][0]:image_roi[1][1]+1,
                                                  image_roi[0][0]:image_roi[0][1]+1],comm=PETSc.COMM_SELF)
     vtkstdDevImage = ConvertNumpyVTKImage(stddevROI[...],"stddev",size_roi,spacing,roiOrigin)
     vtkstdDevWriter = vtk.vtkXMLImageDataWriter()
     vtkstdDevWriter.SetFileName( "stddevROI%d.%04d.%04d.vti" % (MeasurementMapNodeSet, kwargs['fileID'],timeID) )
     vtkstdDevWriter.SetInput( vtkstdDevImage )
     vtkstdDevWriter.Update()

     # set UpdatePrediction = False to study model propagation only
     UpdatePrediction = False
     UpdatePrediction = True
     if( UpdatePrediction ):
       # update the state vector and covariance 
       kalmanGainSolver = "petsc"
       kalmanGainSolver = "superlu_dist"
       kalmanFilter.StateUpdate( kalmanGainSolver )
       kalmanFilter.CovarianceUpdate()
  
     CorrectionStage.pop() # log correction stage
  
     # save prefem to disk for dbg
     UpdateFEMsoln = eqnSystems.GetPetscFEMSystemSolnSubVector( "StateSystem", 0)
     updateFEM = kalmanFilter.ProjectMeasurementMatrix(UpdateFEMsoln)

     vtkUpdateFEMImage = ConvertNumpyVTKImage(updateFEM[...],"updatefem",size_roi,spacing,roiOrigin)
     vtkUpdateFEMWriter = vtk.vtkXMLImageDataWriter()
     vtkUpdateFEMWriter.SetFileName( "updatefemROI%d.%04d.%04d.vti" % (MeasurementMapNodeSet, kwargs['fileID'],timeID) )
     vtkUpdateFEMWriter.SetInput( vtkUpdateFEMImage )
     vtkUpdateFEMWriter.Update()

     # get covariance diagonal and and covariance entries for plotting
     for column in CovCol:
       kalmanFilter.ExtractCoVarianceForPlotting("Cov%04d" % column,column)
     kalmanFilter.ExtractVarianceForPlotting("StateStdDev")
  
     # compute l2 norm of difference
     print femLibrary.WeightedL2Norm(
                         kalmanFilter.systems["StateSystem"], "u0",
                           kalmanFilter.systems["MRTIMean"], "u0*",
                         kalmanFilter.systems["MRTIStdDev"],"du0*") 
     print femLibrary.WeightedL2Norm(
                         kalmanFilter.systems["StateSystem"], "u0",
                           kalmanFilter.systems["MRTIMean"], "u0*",
                        kalmanFilter.systems["StateStdDev"], "du0") 
     #eqnSystems.StoreTransientSystemTimeStep("StateSystem",timeID ) 
     exodusII_IO.WriteTimeStep(MeshOutputFile,eqnSystems, timeID+1, timeID*deltat )  
  retval = dict([])
  retval['fns'] = [0.0]
  retval['rank'] = petscRank 
  return(retval)
# end def kalmanFilter(**kwargs):
########################################################################################

# ----------------------------
# Parse DAKOTA parameters file
# ----------------------------

# setup regular expressions for parameter/label matching
e = '-?(?:\\d+\\.?\\d*|\\.\\d+)[eEdD](?:\\+|-)?\\d+' # exponential notation
f = '-?\\d+\\.\\d*|-?\\.\\d+'                        # floating point
i = '-?\\d+'                                         # integer
value = e+'|'+f+'|'+i                                # numeric field
tag = '\\w+(?::\\w+)*'                               # text tag field

# regular expression for aprepro parameters format
aprepro_regex = re.compile('^\s*\{\s*(' + tag + ')\s*=\s*(' + value +')\s*\}$')
# regular expression for standard parameters format
standard_regex = re.compile('^\s*(' + value +')\s+(' + tag + ')$')

# open DAKOTA parameters file for reading
paramsfile = open(sys.argv[1], 'r')
fileID = int(sys.argv[1].split(".").pop())

# extract the parameters from the file and store in a dictionary
paramsdict = {}
for line in paramsfile:
    m = aprepro_regex.match(line)
    if m:
        paramsdict[m.group(1)] = m.group(2)
    else:
        m = standard_regex.match(line)
        if m:
            paramsdict[m.group(2)] = m.group(1)

paramsfile.close()

# crude error checking; handle both standard and aprepro cases
num_vars = 0
if ('variables' in paramsdict):
    num_vars = int(paramsdict['variables'])
elif ('DAKOTA_VARS' in paramsdict):
    num_vars = int(paramsdict['DAKOTA_VARS'])

num_fns = 0
if ('functions' in paramsdict):
    num_fns = int(paramsdict['functions'])
elif ('DAKOTA_FNS' in paramsdict):
    num_fns = int(paramsdict['DAKOTA_FNS'])

# -------------------------------
# Convert and send to application
# -------------------------------

# set up the data structures the rosenbrock analysis code expects
# for this simple example, put all the variables into a single hardwired array
continuous_vars = { 
                    'modelcov':paramsdict['modelcov'],
                     'uniform':int(paramsdict['uniform']),
                     'algebra':int(paramsdict['algebra']),
                         'roi':int(paramsdict['roi'])
                  }

active_set_vector = [ int(paramsdict['ASV_%d:obj_fn' % (i) ]) for i in range(1,num_fns+1)  ] 
# set a dictionary for passing to rosenbrock via Python kwargs
fem_params              = {}
fem_params['cv']        = continuous_vars
fem_params['asv']       = active_set_vector
fem_params['functions'] = num_fns
fem_params['fileID']    = fileID 

# execute the rosenbrock analysis as a separate Python module
print "Running Kalman..."
fem_results = KalmanFilterMRTI(**fem_params)
print "Kalman complete."

# ----------------------------
# Return the results to DAKOTA
# ----------------------------

if (fem_results['rank'] == 0 ):
  # write the results.out file for return to DAKOTA
  # this example only has a single function, so make some assumptions;
  # not processing DVV
  outfile = open('results.out.tmp.%d' % fileID, 'w')
  
  # write functions
  for func_ind in range(0, num_fns):
      if (active_set_vector[func_ind] & 1):
          functions = fem_results['fns']    
          outfile.write(str(functions[func_ind]) + ' f' + str(func_ind) + '\n')
  
  ## write gradients
  #for func_ind in range(0, num_fns):
  #    if (active_set_vector[func_ind] & 2):
  #        grad = rosen_results['fnGrads'][func_ind]
  #        outfile.write('[ ')
  #        for deriv in grad: 
  #            outfile.write(str(deriv) + ' ')
  #        outfile.write(']\n')
  #
  ## write Hessians
  #for func_ind in range(0, num_fns):
  #    if (active_set_vector[func_ind] & 4):
  #        hessian = rosen_results['fnHessians'][func_ind]
  #        outfile.write('[[ ')
  #        for hessrow in hessian:
  #            for hesscol in hessrow:
  #                outfile.write(str(hesscol) + ' ')
  #            outfile.write('\n')
  #        outfile.write(']]')
  #
  outfile.close();outfile.flush
  #
  # move the temporary results file to the one DAKOTA expects
  import shutil
  shutil.move('results.out.tmp.%d' % fileID, sys.argv[2])
  #os.system('mv results.out.tmp ' + sys.argv[2])
