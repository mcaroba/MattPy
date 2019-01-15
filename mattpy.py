##################################################################################
##################################################################################
#####                                                                        #####
###                                                                            ###
#    @@@@@      @@@@@                 @@@@      @@@     @@@@@@@@@@               # 
#    @@@@@@   *@@@@@&      .,,        @@@.     @@@@     @@@#   @@@@              # 
#   (@@@.@@  &@@ @@@    @@@@@@@@@   @@@@@@@@/@@@@@@@@@ .@@@    &@@@  @@@.   *@@@ # 
#   @@@@ @@@@@@ %@@@          @@@#   @@@#     %@@@     @@@@@@@@@@@    @@@  #@@@  # 
#   @@@  ,@@@@  @@@/   @@@@@@@@@@    @@@      @@@%     @@@&%%#*       @@@ (@@@   # 
#  &@@@   @@.   @@@   @@@%   @@@@   #@@@      @@@     &@@@             @@@@@@    # 
#  @@@,        @@@@   /@@@@@&@@@*    @@@@@%   @@@@@@  @@@@             @@@@.     # 
#                                                                      @@@       # 
#                                                                   @@@@@        #                    
#                                                                                #
#                               MattPy v0.2                                      #
#                                                                                #
#     The following distribution of Python functions for material tensor         #
#         analysis, collectively known as MattPy, has been written by            #
#                                                                                #
#                              Miguel A. Caro                                    #
#               Dept. of Electrical Engineering and Automation                   #
#                     Aalto University, Espoo, Finland                           #
#                             mcaroba@gmail.com                                  #
#                                                                                #
#    They are provided for free, in the hope that they will be useful, but       #
#           with no warranty whatsoever, under the Creative Commons              #
#               Attribution-NonCommercial-ShareAlike license                     #
#             http://creativecommons.org/licenses/by-nc-sa/3.0/                  #
#                                                                                #
#   When publishing work that makes use of the present distribution please       #
#                         have a look and cite                                   #
#                                                                                #
#                              Miguel A. Caro                                    #
#     "Extended scheme for the projection of material tensors of arbitrary       #
#                 symmetry onto a higher symmetry tensor"                        #
#                            arXiv:1408.1219                                     #
#                                                                                #
#     For an in-depth account of the theory underlying the tensor projector      #
#       scheme, please read (and cite, as appropriate) the original work:        #
#                                                                                #
#                     Maher Moakher and Andrew N. Norris                         #
#      "The Closest Elastic Tensor of Arbitrary Symmetry to an Elasticity        #
#                        Tensor of Lower Symmetry"                               #
#                    Journal of Elasticity 85, 215 (2006)                        #
#                                                                                #
###               Distribution last updated on 12 Jan. 2019                    ###
#####                                                                        #####
##################################################################################
##################################################################################



# Load dependencies (some functions might also require scipy, which is then
# loaded inside the function definition)
import numpy as np


##################################################################################
##################################################################################
##### Create the Tensor class and define some basic functions                #####
##################################################################################
##################################################################################
# Tensor class
class Tensor:
# Initialization
 def __init__(self, tensor, form = None, normalized = False, verbose = True):
  self.verbose = verbose
  self.normalized = normalized
  shape = check_shape(tensor, verbose)
  self.shape = shape
# Process a piezoelectric tensor
  if shape[0] == "piezoelectric":
   if not form or form not in ["e", "d"]:
    print_no_form_warning(verbose)
    form = "e"
   if shape[1] == "voigt":
    voigt = symmetrize_tensor(tensor, shape, verbose)
    cartesian = pz_voigt_to_cartesian(voigt, form)
    vector = vectorize_pz_voigt(voigt, form)
   if shape[1] == "cartesian":
    cartesian = symmetrize_tensor(tensor, shape, verbose)
    voigt = pz_cartesian_to_voigt(cartesian, form)
    vector = vectorize_pz_voigt(voigt, form)
   if shape[1] == "vector":
    if not normalized:
     vector = normalize_pz_vector(tensor, form)
    else:
     vector = tensor
    voigt = tensorize_pz_voigt(vector, form)
    cartesian = pz_voigt_to_cartesian(voigt, form)
   components = get_components(voigt, shape)
# Process an elastic tensor
  if shape[0] == "elastic":
   if shape[1] == "voigt":
    voigt = symmetrize_tensor(tensor, shape, verbose)
    cartesian = ela_voigt_to_cartesian(voigt)
    vector = vectorize_ela_voigt(voigt)
   if shape[1] == "cartesian":
    cartesian = symmetrize_tensor(tensor, shape, verbose)
    voigt = ela_cartesian_to_voigt(cartesian)
    vector = vectorize_ela_voigt(voigt)
   if shape[1] == "vector":
    if not normalized:
     vector = normalize_ela_vector(tensor)
    else:
     vector = tensor
    voigt = tensorize_ela_voigt(vector)
    cartesian = ela_voigt_to_cartesian(voigt)
   components = get_components(voigt, shape)
# Process a lattice matrix
  if shape[0] == "lattice":
   if shape[1] == "cartesian":
    cartesian = np.array(tensor)
    voigt = None
    vector = cartesian.flatten()
    components = cartesian.flatten()
# Pass values to self
  self.form = form
  self.vector = vector
  self.voigt = voigt
  self.cartesian = cartesian
  self.components = components
# Define intrinsic methods
# Rotate method
 def rotate(self, angles):
  form = self.form
  shape = self.shape
  if shape[0] == "piezoelectric":
   cartesian = rotate_pz(self.cartesian, angles)
   voigt = pz_cartesian_to_voigt(cartesian, form)
   vector = vectorize_pz_voigt(voigt, form)
   components = get_components(voigt, shape)
  if shape[0] == "elastic":
   cartesian = rotate_ela(self.cartesian, angles)
   voigt = ela_cartesian_to_voigt(cartesian)
   vector = vectorize_ela_voigt(voigt)
   components = get_components(voigt, shape)
  if shape[0] == "lattice":
   cartesian = rotate_lat(self.cartesian, angles)
   voigt = None
   vector = cartesian.flatten()
   components = cartesian.flatten()
  self.vector = vector
  self.voigt = voigt
  self.cartesian = cartesian
  self.components = components
# Project method
 def get_projection(self, sym = None, shapeout = None, verbose = None):
  if verbose == None:
   verbose = self.verbose
  shape = self.shape
  normalized = self.normalized
  form = self.form
  if shapeout == None:
   if shape[1] == "vector" and not normalized:
    shapeout = "components"
   else:
    shapeout = shape[1]
  if shape[0] == "piezoelectric":
   proj = project_pz(self.vector, sym, verbose)
   vector = []
   for i in range(0,18):
    vector.append(proj[i])
   voigt = tensorize_pz_voigt(vector, form)
   components = get_components(voigt, shape)
   cartesian = pz_voigt_to_cartesian(voigt, form)
  if shape[0] == "elastic":
   proj = project_ela(self.vector, sym, verbose)
   vector = []
   for i in range(0,21):
    vector.append(proj[i])
   voigt = tensorize_ela_voigt(vector)
   components = get_components(voigt, shape)
   cartesian = ela_voigt_to_cartesian(voigt)
  if shape[0] == "lattice":
   proj = project_lat(self.vector, sym, verbose)
   vector = []
   for i in range(0,9):
    vector.append(proj[i])
   vector = np.array(vector)
   voigt = None
   components = vector.copy()
   cartesian = lat_components_to_cartesian(components)
  if shapeout == "vector":
   return vector
  if shapeout == "components":
   return components
  if shapeout == "voigt":
   return voigt
  if shapeout == "cartesian":
   return cartesian
# Distances method
 def get_distances(self, form = None, symlist = None,
                   rotate = False, xtol = 1e-8, verbose = None, printmin = False):
  if verbose == None:
   verbose = self.verbose
  if form == None:
   form = self.form
  shape = self.shape
  if symlist == None:
   if shape[0] == "piezoelectric":
    symlist = ["432", "-43m", "6", "-6", "622", "6mm", "-62m", "3", "32",
               "3m", "-4", "-42m", "2", "222", "m", "-2", "mm2", "1"]
   if shape[0] == "elastic":
    symlist = ["iso", "cub", "hex", "3", "32", "4", "4mm", "ort", "mon"]
   if shape[0] == "lattice":
    symlist = ["hex"]
  if shape[0] == "piezoelectric":
   return pz_dist(self.voigt, form, symlist, rotate, xtol, verbose, printmin)
  if shape[0] == "elastic":
   return ela_dist(self.voigt, symlist, rotate, xtol, verbose, printmin)
  if shape[0] == "lattice":
   return lat_dist(self.vector, symlist, rotate, xtol, verbose, printmin)
##################################################################################
# Check the shape passed to the Tensor class
def check_shape(tensor, verbose = True):
 shape = None
 error = False
 try:
  level1 = len(tensor)
 except:
  level1 = 0
 try:
  level2 = len(tensor[0])
 except:
  level2 = 0
 try:
  level3 = len(tensor[0][0])
 except:
  level3 = 0
 try:
  level4 = len(tensor[0][0][0])
 except:
  level4 = 0
# Cartesian elastic
 if level1 == 3 and level2 == 3 and level3 == 3 and level4 == 3:
  for i in range(0,3):
   try:
    dim = len(tensor[i])
    if dim > 3 or dim < 3:
     error = True
     break
   except:
    error = True
    break
   for j in range(0,3):
    try:
     dim = len(tensor[i][j])
     if dim > 3 or dim < 3:
      error = True
      break
    except:
     error = True
     break
    for k in range(0,3):
     try:
      dim = len(tensor[i][j][k])
      if dim > 3 or dim < 3:
       error = True
       break
     except:
      error = True
      break
     for l in range(0,3):
#     Check that all the elements are numbers and that dimensions
#     are consistent
      try:
       tensor[i][j][k][l] += 0
      except:
       error = True
       break
  shape = ["elastic", "cartesian"]
# Voigt elastic
 if level1 == 6 and level2 == 6 and level3 == 0 and level4 == 0:
  for i in range(0,6):
   try:
    dim = len(tensor[i])
    if dim > 6 or dim < 6:
     error = True
     break
   except:
    error = True
    break
   for j in range(0,6):
#   Check that all the elements are numbers and that dimensions
#   are consistent
    try:
     tensor[i][j] += 0
    except:
     error = True
     break
  shape = ["elastic", "voigt"]
# Vector elastic
 if level1 == 21 and level2 == 0 and level3 == 0 and level4 == 0:
  for i in range(0,21):
#  Check that all the elements are numbers and that dimensions
#  are consistent
   try:
    tensor[i] += 0
   except:
    error = True
    break
  shape = ["elastic", "vector"]
# Cartesian piezoelectric
 if level1 == 3 and level2 == 3 and level3 == 3 and level4 == 0:
  for i in range(0,3):
   try:
    dim = len(tensor[i])
    if dim > 3 or dim < 3:
     error = True
     break
   except:
    error = True
    break
   for j in range(0,3):
    try:
     dim = len(tensor[i][j])
     if dim > 3 or dim < 3:
      error = True
      break
    except:
     error = True
     break
    for k in range(0,3):
#    Check that all the elements are numbers and that dimensions
#    are consistent
     try:
      tensor[i][j][k] += 0
     except:
      error = True
      break
  shape = ["piezoelectric", "cartesian"]
# Voigt piezoelectric
 if level1 == 3 and level2 == 6 and level3 == 0 and level4 == 0:
  for i in range(0,3):
   try:
    dim = len(tensor[i])
    if dim > 6 or dim < 6:
     error = True
     break
   except:
    error = True
    break
   for j in range(0,6):
#   Check that all the elements are numbers and that dimensions
#   are consistent
    try:
     tensor[i][j] += 0
    except:
     error = True
     break
  shape = ["piezoelectric", "voigt"]
# Vector piezoelectric
 if level1 == 18 and level2 == 0 and level3 == 0 and level4 == 0:
  for i in range(0,18):
#  Check that all the elements are numbers and that dimensions
#  are consistent
   try:
    tensor[i] += 0
   except:
    error = True
    break
  shape = ["piezoelectric", "vector"]
# Cartesian lattice
 if level1 == 3 and level2 == 3 and level3 == 0 and level4 == 0:
  for i in range(0,3):
   try:
    dim = len(tensor[i])
    if dim > 3 or dim < 3:
     error = True
     break
   except:
    error = True
    break
   for j in range(0,3):
#   Check that all the elements are numbers and that dimensions
#   are consistent
    try:
     tensor[i][j] += 0
    except:
     error = True
     break
  shape = ["lattice", "cartesian"]
# If an error was raised print error message
 if error or not shape:
  print_check_shape_error(verbose)
# Return shape, if not recognized it will be None
 return shape
##################################################################################
def symmetrize_tensor(tensor, shape, verbose):
 flag = False
 if shape[0] == "piezoelectric" and shape[1] == "cartesian":
  for i in range(0,3):
   for j in range(0,3):
    for k in range(j,3):
     if np.abs(tensor[i][j][k] - tensor[i][k][j]) > 0.0001:
      flag=True
     temp = 0.5 * (tensor[i][j][k] + tensor[i][k][j])
     tensor[i][j][k] = temp
     tensor[i][k][j] = temp
 if shape[0] == "elastic" and shape[1] == "cartesian":
  for i in range(0,3):
   for j in range(i,3):
    for k in range(i,3):
     if k > i:
      l0 = k
     else:
      l0 = j
     for l in range(l0,3):
      if np.abs(tensor[i][j][k][l] - tensor[i][j][l][k]) > 0.0001 or \
         np.abs(tensor[i][j][k][l] - tensor[j][i][k][l]) > 0.0001 or \
         np.abs(tensor[i][j][k][l] - tensor[j][i][l][k]) > 0.0001 or \
         np.abs(tensor[i][j][k][l] - tensor[k][l][i][j]) > 0.0001 or \
         np.abs(tensor[i][j][k][l] - tensor[k][l][j][i]) > 0.0001 or \
         np.abs(tensor[i][j][k][l] - tensor[l][k][i][j]) > 0.0001 or \
         np.abs(tensor[i][j][k][l] - tensor[l][k][j][i]) > 0.0001:
       flag=True
      temp = 0.125 * (tensor[i][j][k][l] + tensor[i][j][l][k] + tensor[j][i][k][l] + tensor[j][i][l][k] +
                      tensor[k][l][i][j] + tensor[l][k][i][j] + tensor[k][l][j][i] + tensor[l][k][j][i])
      tensor[i][j][k][l] = temp
      tensor[i][j][l][k] = temp
      tensor[j][i][k][l] = temp
      tensor[j][i][l][k] = temp
      tensor[k][l][i][j] = temp
      tensor[l][k][i][j] = temp
      tensor[k][l][j][i] = temp
      tensor[l][k][j][i] = temp
 if shape[0] == "elastic" and shape[1] == "voigt":
  for i in range(0,6):
   for j in range(i,6):
    if np.abs(tensor[i][j] - tensor[j][i]) > 0.0001:
     flag=True
    temp = 0.5 * (tensor[i][j] + tensor[j][i])
    tensor[i][j] = temp
    tensor[j][i] = temp
 if shape[0] == "piezoelectric" and flag:
  print_pz_tensor_not_symmetric(verbose)
 if shape[0] == "elastic" and flag:
  print_ela_tensor_not_symmetric(verbose)
 return tensor
##################################################################################
def normalize_pz_vector(tensor, form):
 level0=[]
 for i in range(0,3):
  for j in range(0,6):
   k=i*6+j
   if j < 3:
    level0.append(tensor[k])
   else:
    if form == "e":
     level0.append(tensor[k]*np.sqrt(2.))
    if form == "d":
     level0.append(tensor[k]/np.sqrt(2.))
 return level0
##################################################################################
def normalize_ela_vector(tensor):
 level0 = []
 for i in range(0,6):
  sumi = 0
  for li in range(0,i+1):
   sumi += li
  for j in range(i,6):
   sumj = 0
   for lj in range(0,j+1):
    sumj += lj
   if j >= i:
    k = i*6 + j - sumi
   else:
    k = j*6 + i - sumj
   coeff = 1.
   if i != j:
    coeff /= np.sqrt(2.)
   if i >= 3:
    coeff /= np.sqrt(2.)
   if j >= 3:
    coeff /= np.sqrt(2.)
   level0.append(tensor[k]/coeff)
 return level0
##################################################################################
def get_components(voigt, shape):
 level0 = []
 if shape[0] == "piezoelectric":
  for i in range(0,3):
   for j in range(0,6):
    level0.append(voigt[i][j])
 if shape[0] == "elastic":
  for i in range(0,6):
   for j in range(i,6):
    level0.append(voigt[i][j])
 return level0
##################################################################################
##################################################################################
##### End of Tensor class and basic functions                                #####
##################################################################################
##################################################################################





##################################################################################
##################################################################################
##### Define some error/warning printing functions. All of these functions   #####
##### take the "verbose" variable as input, so that the messages can be      #####
##### switched off                                                           #####
##################################################################################
##################################################################################
# Prints an error if there is an attempt to initialize the Tensor class with
# a list with the wrong shape
def print_check_shape_error(verbose):
 if verbose:
  print "                                                                   "
  print "**************************** E R R O R ****************************"
  print "The material tensor you have defined has an unknown shape. Please  "
  print "check that the dimensions are compatible with the acceptable ones: "
  print "                                                                   "
  print " Elastic: 3x3x3x3 (Cartesian), 6x6 (Voigt), 21 (vector)            "
  print "                                                                   "
  print " Piezoelectric: 3x3x3 (Cartesian), 3x6 (Voigt), 18 (vector)        "
  print "**************************** E R R O R ****************************"
  print "                                                                   "
##################################################################################
def print_no_form_warning(verbose):
 if verbose:
  print "                                                                   "
  print "************************** W A R N I N G **************************"
  print "Warning! You have not defined a form (keyword \"form\") for your   "
  print "piezoelectric tensor, I'm using e_ij by default (form = \"e\").    "
  print "You can also use the d_ij by specifying form = \"d\". Both forms   "
  print "make use of the same projectors for all the piezoelectric point    "
  print "groups but have different normalizing factors in vector            "
  print "representation that need to be accounted for.                      "
  print "************************** W A R N I N G **************************"
  print "                                                                   "
##################################################################################
def print_pz_tensor_not_symmetric(verbose):
 if verbose:
  print "                                                                   "
  print "************************** W A R N I N G **************************"
  print "Warning! Your piezo tensor is not symmetric, I'm symmetrizing it!  "
  print "************************** W A R N I N G **************************"
  print "                                                                   "
##################################################################################
def print_ela_tensor_not_symmetric(verbose):
 if verbose:
  print "                                                                   "
  print "************************** W A R N I N G **************************"
  print "Warning! Your elastic tensor is not symmetric, I'm symmetrizing it!"
  print "************************** W A R N I N G **************************"
  print "                                                                   "
##################################################################################
##################################################################################
##### End of printing functions                                              #####
##################################################################################
##################################################################################





##################################################################################
##################################################################################
##### All the functions for manipulation of lattice matrices are below       #####
##################################################################################
##################################################################################
##################################################################################
# Performs a rotation operation on a (Cartesian) rank-3 tensor
def rotate_lat(e_cart,rot_angles):
 f = np.pi / 180.
 result=np.zeros((3,3))
 tx=f*rot_angles[0] ; ty=f*rot_angles[1] ; tz=f*rot_angles[2]
 Rx=[[1., 0., 0.], [0., np.cos(tx), 0.-np.sin(tx)], [0., np.sin(tx), np.cos(tx)]]
 Ry=[[np.cos(ty), 0., np.sin(ty)], [0., 1., 0.], [0.-np.sin(ty), 0., np.cos(ty)]]
 Rz=[[np.cos(tz), 0.-np.sin(tz), 0.], [np.sin(tz), np.cos(tz), 0.], [0., 0., 1.]]
 R=np.dot(Rz,np.dot(Ry,Rx))
 for i in range(0,3):
  for j in range(0,3):
   temp = 0.
   for m in range(0,3):
    for n in range(0,3):
     temp += R[i][m]*R[j][n]*e_cart[m][n]
   result[i][j] = temp
 return result
##################################################################################
# Transforms from a flat array of components to a 3x3 cartesian representation
def lat_components_to_cartesian(e_components):
 e_cart = np.zeros([3,3])
 k = 0
 for i in range(0,3):
  for j in range(0,3):
   e_cart[i][j] = e_components[k]
   k += 1
 return e_cart
##################################################################################
# Projects onto a given reference lattice
def project_lat(vector, sym = None, verbose = True):
# Available classes and point groups ("iso" does not apply here)
 classes = ["cub", "hex", "hex60", "rho", "tig", "tet", "ort", "mon", "tic"]
 pointgroups = ["23", "m-3", "432", "-43m", "m-3m", "6", "-6", "6/m",
                "622", "6mm", "-62m", "6/mmm", "3", "-3", "32", "3m",
                "-3m", "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
                "2", "2/m", "222", "m", "-2", "mm2", "mmm", "1", "-1"]
# Default to "cub" if sym is not defined and print warning (warning can
# be switched off with verbose = False)
 if not sym:
  sym = "cub"
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! You have not defined a symmetry, using cubic lattice     !"
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# Print warning and default to "cub" if symmetry is not on the list
 if sym not in classes:
  sym = "cub"
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! I could not understand the symmetry you have defined,     "
   print "using cubic lattice instead! The list of available symmetries      "
   print "from which you have to choose (\"sym\" keyword) is:                "
   print "Crystal classes:                                                   "
   print classes
   print "Point groups:                                                      "
   print pointgroups
   print "                                                                   "
   print "Note that hexagonal lattices can be defined with angles of either  "
   print "120 degrees (canonical representation, use \"hex\" or any hexagonal  "
   print "point group), 60 degrees (use \"hex60\"), or in rhombohedral         "
   print "representation (use \"rho\")                                         "
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# If user does not give a point group (but a class instead) then a default
# point group compatible with that class will be assigned when the class
# has more than one independent form for the elastic tensor (i.e. the two
# forms differ by more than modulo a rotation). We make this opaque to the
# user for lattice projections.
 defaultpg = {"tig": "3", "tet": "4"}
 if defaultpg.get(sym):
  oldsym = sym
  sym = defaultpg[oldsym]
  if 0:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! You have chosen a crystal class (", oldsym, ") with more  "
   print "than one independent form of the elastic tensor! I am defaulting to"
   print "point group", defaultpg[oldsym], ".                                "
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# Initialize projector
 projector=np.zeros((9,9))
# Obtain matrix elements <----------------------- FIX THIS, I NEED TO ADD ALL THE LATTICE SYSTEMS WITH MATHEMATICA
# Cubic
 if sym == "cub" or sym == "23" or sym == "m-3" or sym == "432" or sym == "-43m" or sym == "m-3m":
  print "Not implemented!"
# Hexagonal
 if sym == "hex" or sym == "6" or sym == "-6" or sym == "6/m" or sym == "622" \
    or sym == "6mm" or sym == "-62m" or sym == "6/mmm":
  c1 = 1./2. ; c2 = -1./4. ; c3 = np.sqrt(3.)/4. ; c4 = 1./8. ; c5 = -np.sqrt(3.)/8.
  c6 = 3./8. ; c7 = 1.
  projector[0][0] = c1 ; projector[0][1] = c2 ; projector[0][4] = c3
  projector[1][0] = c2 ; projector[1][1] = c4 ; projector[1][4] = c5
  projector[4][0] = c3 ; projector[4][1] = c5 ; projector[4][4] = c6
  projector[8][8] = c7
 if sym == "rho":
  print "Not implemented!"
# Trigonal (point groups 3 and -3)
 if sym == "3" or sym == "-3":
  print "Not implemented!"
# Trigonal (point groups 32, 3m and -3m)
 if sym == "32" or sym == "3m" or sym == "-3m":
  print "Not implemented!"
# Tetragonal (point groups 4, -4, 4/m)
 if sym == "4" or sym == "-4" or sym == "4/m":
  print "Not implemented!"
# Tetragonal (point groups 422, 4mm, -42m, 4/mmm)
 if sym == "422" or sym == "4mm" or sym == "-42m" or sym == "4/mmm":
  print "Not implemented!"
# Orthorhombic
 if sym == "ort" or sym == "222" or sym == "mm2" or sym == "mmm":
  print "Not implemented!"
# Monoclinic
 if sym == "mon" or sym == "2" or sym == "2/m" or sym == "m" or sym == "-2":
  print "Not implemented!"
# Triclinic
 if sym == "tic" or sym == "1" or sym == "-1":
  c1 = 1.
  for i in range(0,9):
   projector[i][i] = c1
# Carry out the projection
 proj=np.dot(projector,vector)
 return proj
##################################################################################
def res_lat(t, vector, sym = None, verbose = False):
 tx=t[0] ; ty=t[1] ; tz=t[2]
 c_cart=lat_components_to_cartesian(vector)
 rot_c=rotate_lat(c_cart,[tx,ty,tz])
 rot_vector=np.array(rot_c).flatten()
 proj_rot_vector=project_lat(rot_vector, sym = sym, verbose = verbose)
 res=rot_vector-proj_rot_vector
 result=np.dot(res,res)
 return result
##################################################################################
# <---------------------------------- FIX THIS. THE SYMLIST SHOULD CONTAIN ALL OF THEM
def lat_dist(vector,
             symlist = ["hex"],
             rotate = False, xtol = 1e-8, verbose = True, printmin = False):
 from scipy.optimize import fmin
 disp = 0
 if printmin:
  disp = 1
 result = []
 if not rotate:
  print "                                                                   "
  print "************************** R E S U L T S **************************"
  print "Results without rotation optimization                              "
  print "                                                                   "
  print "Symmetry     Euclidean distance                                    "
  print "--------     ------------------                                    "
  for sym in symlist:
   v = vector.copy()
   vp = project_lat(v, sym)
   edist2 = np.dot(v-vp,v-vp)
   edist = np.sqrt(edist2)
   print "%8s         %7.4f Angst." % (sym, edist)
   result.append([sym, edist])
  print "************************** R E S U L T S **************************"
  print "                                                                   "
 if rotate:
  print "                                                                   "
  print "************************** R E S U L T S **************************"
  print "Results with rotation optimization                                 "
  print "                                                                   "
  print "Symmetry     Euclidean distance     Angles tx,     ty,     tz      "
  print "--------     ------------------     -------------------------------"
  for sym in symlist:
   topt = [0., 0., 0.]
   topt = fmin(res_lat, x0=[0,0,0], xtol=xtol, args=(vector, sym, verbose), disp=disp)
   ct = lat_components_to_cartesian(vector)
   rotct = rotate_lat(ct, topt)
   v = np.array(rotct).flatten()
   vp = project_lat(v, sym = sym, verbose=False)
   edist2 = np.dot(v-vp,v-vp)
   edist = np.sqrt(edist2)
   printangles = ["%7.2f" % topt[0], "%7.2f" % topt[1], "%7.2f" % topt[2]]
   print "%8s         %7.2f Angst.       %s %s %s  deg." \
         % (sym, edist, printangles[0], printangles[1], printangles[2])
   result.append([sym, edist, topt[0], topt[1], topt[2]])
  print "************************** R E S U L T S **************************"
  print "                                                                   "
 return result
##################################################################################
##################################################################################
##################################################################################
##### End of functions for lattice matrix manipulation                       #####
##################################################################################
##################################################################################





##################################################################################
##################################################################################
##### All the functions for manipulation of piezoelectric tensors are below  #####
##################################################################################
##################################################################################
# Turns PZ tensor in Voigt notation to vector (preserving the norm)
# d_ij and e_ij forms have a different vector representation
def vectorize_pz_voigt(e_voigt, form):
 result=[]
 for i in range(0,3):
  for j in range(0,6):
   if j < 3:
    result.append(e_voigt[i][j])
   else:
    if form == "e":
     result.append(np.sqrt(2.)*e_voigt[i][j])
    if form == "d":
     result.append(e_voigt[i][j]/np.sqrt(2.))
 return result
##################################################################################
# Turns PZ vector (assumed to preserve the norm) to tensor in Voigt notation
def tensorize_pz_voigt(vector_e_voigt, form):
 level0=[]
 for i in range(0,3):
  level1=[]
  for j in range(0,6):
   k=i*6+j
   if j < 3:
    level1.append(vector_e_voigt[k])
   else:
    if form == "e":
     level1.append(vector_e_voigt[k]/np.sqrt(2.))
    if form == "d":
     level1.append(vector_e_voigt[k]*np.sqrt(2.))
  level0.append(level1)
 return level0
##################################################################################
# Transforms PZ tensor in Voigt notation to Cartesian notation
def pz_voigt_to_cartesian(e_voigt, form):
 level0=[]
 for i in range(0,3):
  level1=[]
  for j in range(0,3):
   level2=[]
   for k in range(0,3):
    i_voigt=i
    if j == k:
     j_voigt=j
    else:
     if (j == 1 and k == 2) or (k == 1 and j == 2): 
      j_voigt=3
     elif (j == 0 and k == 2) or (k == 0 and j == 2):
      j_voigt=4
     elif (j == 0 and k == 1) or (k == 0 and j == 1):
      j_voigt=5
    if form == "e":
     level2.append(e_voigt[i_voigt][j_voigt])
    if form == "d":
     level2.append(e_voigt[i_voigt][j_voigt]/2.)
   level1.append(level2)
  level0.append(level1)
 return level0
##################################################################################
# Transforms PZ tensor in Cartesian notation to Voigt notation
def pz_cartesian_to_voigt(e_cart, form):
 level0=[]
 for i_voigt in range(0,3):
  level1=[]
  for j_voigt in range(0,6):
    i=i_voigt
    if j_voigt < 3:
     j=j_voigt
     k=j_voigt
    else:
     if j_voigt == 3:
      j=1 ; k=2
     elif j_voigt == 4:
      j=0 ; k=2
     elif j_voigt == 5:
      j=0 ; k=1
    if form == "e":
     level1.append(e_cart[i][j][k])
    if form == "d":
     level1.append(2.*e_cart[i][j][k])
  level0.append(level1)
 return level0
##################################################################################
# Performs a rotation operation on a (Cartesian) rank-3 tensor
def rotate_pz(e_cart,rot_angles):
 f = np.pi / 180.
 result=np.zeros((3,3,3))
 tx=f*rot_angles[0] ; ty=f*rot_angles[1] ; tz=f*rot_angles[2]
 Rx=[[1., 0., 0.], [0., np.cos(tx), 0.-np.sin(tx)], [0., np.sin(tx), np.cos(tx)]]
 Ry=[[np.cos(ty), 0., np.sin(ty)], [0., 1., 0.], [0.-np.sin(ty), 0., np.cos(ty)]]
 Rz=[[np.cos(tz), 0.-np.sin(tz), 0.], [np.sin(tz), np.cos(tz), 0.], [0., 0., 1.]]
 R=np.dot(Rz,np.dot(Ry,Rx))
 for i in range(0,3):
  for j in range(0,3):
   for k in range(0,3):
    temp = 0.
    for m in range(0,3):
     for n in range(0,3):
      for o in range(0,3):
       temp += R[i][m]*R[j][n]*R[k][o]*e_cart[m][n][o]
    result[i][j][k]=temp
 return result
##################################################################################
# Projects onto a piezoelectric tensor (tensor in vector form)
def project_pz(vector_e_voigt, sym = None, verbose = True):
# Available classes, non centrosymmetric point groups and centrosymmetric point groups
 classes = ["iso", "cub", "hex", "tig", "tet", "ort", "mon", "tic"]
 ncspointgroups = ["23", "432", "-43m", "6", "-6",
                   "622", "6mm", "-62m", "3", "32", "3m",
                   "4", "-4", "422", "4mm", "-42m",
                   "2", "222", "m", "-2", "mm2", "1"]
 cspointgroups = ["m-3", "m-3m", "6/m",
                  "6/mmm", "-3",
                  "-3m", "4/m", "4/mmm",
                  "2/m", "mmm", "-1"]
 pointgroups = ncspointgroups + cspointgroups
# Print warning if user chooses a centrosymmetric point group or isotropy
 if sym in cspointgroups:
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! You have chosen a centrosymmetric point group, the        "
   print "projection will be zero!                                           "
   print "************************** W A R N I N G **************************"
   print "                                                                   "
 if sym == "iso":
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! You have chosen material isotropy, the projection will    "
   print "be zero!                                                           "
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# Default to "-43m" if sym is not defined and print warning (warning can
# be switched off with verbose = False)
 if not sym:
  sym = "-43m"
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! You have not defined a symmetry, using PG -43m tensor!    "
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# Print warning and default to "-43m" if symmetry is not on the list
 if sym not in classes and sym not in pointgroups:
  sym = "-43m"
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! I could not understand the symmetry you have defined,     "
   print "using PG -43m tensor instead! The list of available symmetries     "
   print "from which you have to choose (\"sym\" keyword) is:                "
   print "Crystal classes:                                                   "
   print classes
   print "Point groups:                                                      "
   print pointgroups
   print "Note! The form of the piezoelectric tensor depends on the specific "
   print "point group, not only the crystal class. If you choose a crystal   "
   print "class I will assign a default point group for that class, which may"
   print "or may not be the one you need to use!                             "
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# If user does not give a point group (but a class instead) then a default
# point group compatible with that class will be assigned when the class
# has more than one independent form for the piezoelectric tensor (i.e. the two
# forms differ by more than modulo a rotation) 
 defaultpg = {"cub": "-43m", "hex": "6mm", "tig": "3m", "tet": "4mm", "ort" :"222", "mon": "2", "tic": "1"}
 if defaultpg.get(sym):
  oldsym = sym
  sym = defaultpg[oldsym]
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! You have chosen a crystal class (", oldsym, ") with more  "
   print "than one independent form of the piezoelectric tensor! I am        "
   print "defaulting to point group", defaultpg[oldsym], ".                  "
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# Initialize projector
 projector=np.zeros((18,18))
# Obtain matrix elements
# Isotropic or centrosymmetric (or PG 432 which does not have first-order piezo), do nothing
 if sym == "iso" or sym in cspointgroups or sym == "432":
  pass
# Cubic
 if sym == "-43m" or sym == "23":
  c1 = 1./3.
  projector[3][3] = c1  ; projector[3][10] = c1  ; projector[3][17] = c1
  projector[10][3] = c1 ; projector[10][10] = c1 ; projector[10][17] = c1
  projector[17][3] = c1 ; projector[17][10] = c1 ; projector[17][17] = c1
  projector[3][3] = c1  ; projector[3][10] = c1  ; projector[3][17] = c1
  projector[10][3] = c1 ; projector[10][10] = c1 ; projector[10][17] = c1
  projector[17][3] = c1 ; projector[17][10] = c1 ; projector[17][17] = c1
# Hexagonal (some tetragonal point groups as well)
 if sym == "6" or sym == "4":
  c1 = 1./2. ; c2 = 1.
  projector[3][3] = c1   ; projector[3][10] = -c1 ; projector[4][4] = c1
  projector[4][9] = c1   ; projector[9][4] = c1   ; projector[9][9] = c1
  projector[10][3] = -c1 ; projector[10][10] = c1 ; projector[12][12] = c1
  projector[12][13] = c1 ; projector[13][12] = c1 ; projector[13][13] = c1
  projector[14][14] = c2
 if sym == "6mm" or sym == "4mm":
  c1 = 1./2. ; c2 = 1.
  projector[4][4] = c1   ; projector[4][9] = c1   ; projector[9][4] = c1
  projector[9][9] = c1   ; projector[12][12] = c1 ; projector[12][13] = c1
  projector[13][12] = c1 ; projector[13][13] = c1 ; projector[14][14] = c2
 if sym == "622" or sym == "422":
  c1 = 1./2.
  projector[3][3] = c1 ; projector[3][10] = -c1 ; projector[10][3] = -c1
  projector[10][10] = c1
 if sym == "-6":
  c1 = 1./4. ; c2 = 1./2./np.sqrt(2.) ; c3 = 1./2.
  projector[0][0] = c1   ; projector[0][1] = -c1 ; projector[0][11] = -c2
  projector[1][0] = -c1  ; projector[1][1] = c1  ; projector[1][11] = c2
  projector[5][5] = c3   ; projector[5][6] = c2  ; projector[5][7] = -c2
  projector[6][5] = c2   ; projector[6][6] = c1  ; projector[6][7] = -c1
  projector[7][5] = -c2  ; projector[7][6] = -c1 ; projector[7][7] = c1
  projector[11][0] = -c2 ; projector[11][1] = c2 ; projector[11][11] = c3
 if sym == "-62m":
  c1 = 1./2. ; c2 = 1./2./np.sqrt(2.) ; c3 = 1./4.
  projector[5][5] = c1  ; projector[5][6] = c2  ; projector[5][7] = -c2
  projector[6][5] = c2  ; projector[6][6] = c3  ; projector[6][7] = -c3
  projector[7][5] = -c2 ; projector[7][6] = -c3 ; projector[7][7] = c3
# Trigonal
 if sym == "3":
  c1 = 1./4. ; c2 = 1./2./np.sqrt(2.) ; c3 = 1./2. ; c4 = 1.
  projector[0][0] = c1   ; projector[0][1] = -c1  ; projector[0][11] = -c2
  projector[1][0] = -c1  ; projector[1][1] = c1   ; projector[1][11] = c2
  projector[3][3] = c3   ; projector[3][10] = -c3 ; projector[4][4] = c3
  projector[4][9] = c3   ; projector[5][5] = c3   ; projector[5][6] = c2
  projector[5][7] = -c2  ; projector[6][5] = c2   ; projector[6][6] = c1
  projector[6][7] = -c1  ; projector[7][5] = -c2  ; projector[7][6] = -c1
  projector[7][7] = c1   ; projector[9][4] = c3   ; projector[9][9] = c3
  projector[10][3] = -c3 ; projector[10][10] = c3 ; projector[11][0] = -c2
  projector[11][1] = c2  ; projector[11][11] = c3 ; projector[12][12] = c3
  projector[12][13] = c3 ; projector[13][12] = c3 ; projector[13][13] = c3
  projector[14][14] = c4
 if sym == "32":
  c1 = 1./4. ; c2 = 1./2./np.sqrt(2.) ; c3 = 1./2.
  projector[0][0] = c1 ; projector[0][1] = -c1 ; projector[0][11] = -c2
  projector[1][0] = -c1 ; projector[1][1] = c1 ; projector[1][11] = c2
  projector[3][3] = c3 ; projector[3][10] = -c3 ; projector[10][3] = -c3
  projector[10][10] = c3 ; projector[11][0] = -c2 ; projector[11][1] = c2
  projector[11][11] = c3
 if sym == "3m":
  c1 = 1./2. ; c2 = 1./2./np.sqrt(2.) ; c3 = 1./4. ; c4 = 1.
  projector[4][4] = c1   ; projector[4][9] = c1   ; projector[5][5] = c1
  projector[5][6] = c2   ; projector[5][7] = -c2  ; projector[6][5] = c2
  projector[6][6] = c3   ; projector[6][7] = -c3  ; projector[7][5] = -c2
  projector[7][6] = -c3  ; projector[7][7] = c3   ; projector[9][4] = c1
  projector[9][9] = c1   ; projector[12][12] = c1 ; projector[12][13] = c1
  projector[13][12] = c1 ; projector[13][13] = c1 ; projector[14][14] = c4
# Tetragonal (not all, some are under hexagonal)
 if sym == "-4":
  c1 = 1./2. ; c2 = 1.
  projector[3][3] = c1 ; projector[3][10] = c1 ; projector[4][4] = c1
  projector[4][9] = -c1 ; projector[9][4] = -c1 ; projector[9][9] = c1
  projector[10][3] = c1 ; projector[10][10] = c1 ; projector[12][12] = c1
  projector[12][13] = -c1 ; projector[13][12] = -c1 ; projector[13][13] = c1
  projector[17][17] = c2
 if sym == "-42m":
  c1 = 1./2. ; c2 = 1.
  projector[3][3] = c1   ; projector[3][10] = c1 ; projector[10][3] = c1
  projector[10][10] = c1 ; projector[17][17] = c2
# Orthorhombic
 if sym == "222":
  c1 = 1.
  projector[3][3] = c1 ; projector[10][10] = c1 ; projector[17][17] = c1
 if sym == "mm2":
  c1 = 1.
  projector[4][4] = c1   ; projector[9][9] = c1 ; projector[12][12] = c1
  projector[13][13] = c1 ; projector[14][14] = c1
# Monoclinic
 if sym == "2":
  c1 = 1.
  projector[3][3] = c1 ; projector[5][5] = c1 ; projector[6][6] = c1
  projector[7][7] = c1 ; projector[8][8] = c1 ; projector[10][10] = c1
  projector[15][15] = c1 ; projector[17][17] = c1
 if sym == "m":
  c1 = 1.
  projector[0][0] = c1 ; projector[1][1] = c1 ; projector[2][2] = c1
  projector[4][4] = c1 ; projector[9][9] = c1 ; projector[11][11] = c1
  projector[12][12] = c1 ; projector[13][13] = c1 ; projector[14][14] = c1
  projector[16][16] = c1
# Triclinic
 if sym == "1":
  c1 = 1.
  for i in range(0,18):
   projector[i][i] = c1
# Carry out the projection
 proj=np.dot(projector,vector_e_voigt)
 return proj
##################################################################################
# Creates the function to be minimized for an input PZ tensor
# given in Voigt notation, in terms of the rotation angles
def res_pz(t, e_voigt, sym = None, form = None, verbose = True):
 tx=t[0] ; ty=t[1] ; tz=t[2]
 e_cart=pz_voigt_to_cartesian(e_voigt, form = form)
 rot_e=rotate_pz(e_cart,[tx,ty,tz])
 rot_e_voigt=pz_cartesian_to_voigt(rot_e, form = form)
 rot_vector=vectorize_pz_voigt(rot_e_voigt, form = form)
 proj_rot_vector=project_pz(rot_vector,sym, verbose = verbose)
 res=rot_vector-proj_rot_vector
 result=np.dot(res,res)
 return result
##################################################################################
# Checks for the possible symmetry projections and gives the Euclidean
# distance for each of them. This allows to find out the most probable underlying
# symmetry of the tensor. Note that the distance will in general be reduced
# as the number of independent piezoelectric constants is allowed to increase (e.g. for
# a point group 1 projection the distance is zero). The function accepts two modes:
# with and without rotation optimization. Setting printmin = True will print
# the info from the minimization routine. The list of symmetries to check is
# complete by default. The user can override this if they're only interested
# in a reduced set. This function requires Scipy.
def pz_dist(e_voigt, form = None,
            symlist = ["432", "-43m", "6", "-6", "622", "6mm", "-62m", "3", "32", "3m",
                       "-4", "-42m", "2", "222", "m", "-2", "mm2", "1"],
            rotate = False, xtol = 1e-8, verbose = True, printmin = False):
 from scipy.optimize import fmin
 cspointgroups = ["m-3", "m-3m", "6/m", "6/mmm", "-3", "-3m", "4/m", "4/mmm", "2/m", "mmm", "-1"]
 disp = 0
 if printmin:
  disp = 1
 if not form:
  form = "e"
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! You have not defined a form (keyword \"form\"), using e_ij"
   print "by default (form = \"e\"). You can also use the d_ij by specifying "
   print "form = \"d\". Both forms use the same projectors for all the       "
   print "piezoelectric point groups but have a different normalizing factors"
   print "in vector representation that need to be accounted for.            "
   print "************************** W A R N I N G **************************"
   print "                                                                   "
 result = []
 if not rotate:
  print "                                                                   "
  print "************************** R E S U L T S **************************"
  print "Results without rotation optimization                              "
  print "                                                                   "
  print "Symmetry     Euclidean distance                                    "
  print "--------     ------------------                                    "
  for sym in symlist:
   v = vectorize_pz_voigt(e_voigt, form = form)
   vp = project_pz(v, sym)
   edist2 = np.dot(v-vp,v-vp)
   edist = np.sqrt(edist2)
   print "%8s          %7.2f C/m^2" % (sym, edist)
   result.append([sym, edist])
  print "************************** R E S U L T S **************************"
  print "                                                                   "
 if rotate:
  print "                                                                   "
  print "************************** R E S U L T S **************************"
  print "Results with rotation optimization                                 "
  print "                                                                   "
  print "Symmetry     Euclidean distance     Angles tx,     ty,     tz      "
  print "--------     ------------------     -------------------------------"
  for sym in symlist:
   topt = [0., 0., 0.]
   if sym != "iso" or sym not in cspointgroups:
    topt = fmin(res_pz, x0=[0,0,0], xtol=xtol, args=(e_voigt, sym, form, verbose), disp=disp)
   et = pz_voigt_to_cartesian(e_voigt, form = form)
   rotet = rotate_pz(et, topt)
   rot_voigt = pz_cartesian_to_voigt(rotet, form = form)
   v = vectorize_pz_voigt(rot_voigt, form = form)
   vp = project_pz(v, sym = sym, verbose=False)
   edist2 = np.dot(v-vp,v-vp)
   edist = np.sqrt(edist2)
   printangles = ["%7.2f" % topt[0], "%7.2f" % topt[1], "%7.2f" % topt[2]]
#  Take symmetry planes into account
   if sym == "iso" or sym in cspointgroups:
    printangles = ["    n/a", "    n/a", "    n/a"]
#  I need to take symmetry planes into account to notify about redundant angles         <--- TO DO
#  At the moment the code commented out below is the correct one for elasticity         <--- TO DO
#   if sym == "hex" or sym == "6" or sym == "-6" or sym == "6/m" or sym == "622" \
#      or sym == "6mm" or sym == "-62m" or sym == "6/mmm" or sym == "3" or sym == "-3" \
#      or sym == "32" or sym == "3m" or sym == "-3m":
#    printangles = ["%7.2f" % topt[0], "%7.2f" % topt[1], "    n/a"]
   print "%8s          %7.2f C/m^2       %s %s %s  deg." \
         % (sym, edist, printangles[0], printangles[1], printangles[2])
   result.append([sym, edist, topt[0], topt[1], topt[2]])
  print "************************** R E S U L T S **************************"
  print "                                                                   "
 return result
##################################################################################
##################################################################################
############# End of functions for piezoelectric tensor manipulation #############
##################################################################################
##################################################################################




##################################################################################
##################################################################################
####### All the functions for manipulation of stiffness tensors are below  #######
##################################################################################
##################################################################################
# Turns elastic tensor in Voigt notation to vector (preserving the norm)
# it also symmetrizes the tensor in case it's not already symmetric
def vectorize_ela_voigt(c_voigt):
 result=[]
 for i in range(0,6):
  for j in range(i,6):
   coeff = 1.
   if i != j:
    coeff *= np.sqrt(2.)
   if i >= 3:
    coeff *= np.sqrt(2.)
   if j >= 3:
    coeff *= np.sqrt(2.)
   result.append(coeff*(c_voigt[i][j]+c_voigt[j][i])/2.)
 return result
##################################################################################
# Turns elastic vector (assumed to preserve the norm) to tensor in Voigt notation
def tensorize_ela_voigt(vector_c_voigt):
 level0 = []
 for i in range(0,6):
  level1=[]
  sumi = 0
  for li in range(0,i+1):
   sumi += li
  for j in range(0,6):
   sumj = 0
   for lj in range(0,j+1):
    sumj += lj
   if j >= i:
    k = i*6 + j - sumi
   else:
    k = j*6 + i - sumj
   coeff = 1.
   if i != j:
    coeff *= np.sqrt(2.)
   if i >= 3:
    coeff *= np.sqrt(2.)
   if j >= 3:
    coeff *= np.sqrt(2.)
   level1.append(vector_c_voigt[k]/coeff)
  level0.append(level1)
 return level0
##################################################################################
# Transforms elastic tensor in Voigt notation to Cartesian notation
def ela_voigt_to_cartesian(c_voigt):
 level0=[]
 for i in range(0,3):
  level1=[]
  for j in range(0,3):
   level2=[]
   if i == j:
    i_voigt=i
   else:
    if (i == 1 and j == 2) or (j == 1 and i == 2): 
     i_voigt=3
    elif (i == 0 and j == 2) or (j == 0 and i == 2):
     i_voigt=4
    elif (i == 0 and j == 1) or (j == 0 and i == 1):
     i_voigt=5
   for k in range(0,3):
    level3=[]
    for l in range(0,3):
     if k == l:
      j_voigt=k
     else:
      if (k == 1 and l == 2) or (l == 1 and k == 2): 
       j_voigt=3
      elif (k == 0 and l == 2) or (l == 0 and k == 2):
       j_voigt=4
      elif (k == 0 and l == 1) or (l == 0 and k == 1):
       j_voigt=5
     level3.append(c_voigt[i_voigt][j_voigt])
    level2.append(level3)
   level1.append(level2)
  level0.append(level1)
 return level0
##################################################################################
# Transforms elastic tensor in Cartesian notation to Voigt notation
def ela_cartesian_to_voigt(c_cart):
 level0=[]
 for i_voigt in range(0,6):
  level1=[]
  if i_voigt < 3:
   i=i_voigt
   j=i_voigt
  else:
   if i_voigt == 3:
    i=1 ; j=2
   elif i_voigt == 4:
    i=0 ; j=2
   elif i_voigt == 5:
    i=0 ; j=1
  for j_voigt in range(0,6):
    if j_voigt < 3:
     k=j_voigt
     l=j_voigt
    else:
     if j_voigt == 3:
      k=1 ; l=2
     elif j_voigt == 4:
      k=0 ; l=2
     elif j_voigt == 5:
      k=0 ; l=1
    level1.append(c_cart[i][j][k][l])
  level0.append(level1)
 return level0
##################################################################################
# Performs a rotation operation on a (Cartesian) rank-4 tensor
def rotate_ela(c_cart, rot_angles):
 f = np.pi / 180.
 result=np.zeros((3,3,3,3))
 tx=f*rot_angles[0] ; ty=f*rot_angles[1] ; tz=f*rot_angles[2]
 Rx=[[1., 0., 0.], [0., np.cos(tx), 0.-np.sin(tx)], [0., np.sin(tx), np.cos(tx)]]
 Ry=[[np.cos(ty), 0., np.sin(ty)], [0., 1., 0.], [0.-np.sin(ty), 0., np.cos(ty)]]
 Rz=[[np.cos(tz), 0.-np.sin(tz), 0.], [np.sin(tz), np.cos(tz), 0.], [0., 0., 1.]]
 R=np.dot(Rz,np.dot(Ry,Rx))
 for i in range(0,3):
  for j in range(0,3):
   for k in range(0,3):
    for l in range(0,3):
     temp = 0.
     for m in range(0,3):
      for n in range(0,3):
       for o in range(0,3):
        for p in range(0,3):
         temp += R[i][m]*R[j][n]*R[k][o]*R[l][p]*c_cart[m][n][o][p]
     result[i][j][k][l]=temp
 return result
##################################################################################
# Projects onto an elastic tensor (tensor in vector form)
def project_ela(vector_c_voigt, sym = None, verbose = True):
# Available classes and point groups
 classes = ["iso", "cub", "hex", "tig", "tet", "ort", "mon", "tic"]
 pointgroups = ["23", "m-3", "432", "-43m", "m-3m", "6", "-6", "6/m",
                "622", "6mm", "-62m", "6/mmm", "3", "-3", "32", "3m",
                "-3m", "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
                "2", "2/m", "222", "m", "-2", "mm2", "mmm", "1", "-1"]
# Default to "iso" if sym is not defined and print warning (warning can
# be switched off with verbose = False)
 if not sym:
  sym = "iso"
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! You have not defined a symmetry, using isotropic tensor  !"
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# Print warning and default to "iso" if symmetry is not on the list
 if sym not in classes and sym not in pointgroups:
  sym = "iso"
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! I could not understand the symmetry you have defined,     "
   print "using isotropic tensor instead! The list of available symmetries   "
   print "from which you have to choose (\"sym\" keyword) is:                "
   print "Crystal classes:                                                   "
   print classes
   print "Point groups:                                                      "
   print pointgroups
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# If user does not give a point group (but a class instead) then a default
# point group compatible with that class will be assigned when the class
# has more than one independent form for the elastic tensor (i.e. the two
# forms differ by more than modulo a rotation) 
 defaultpg = {"tig": "3", "tet": "4"}
 if defaultpg.get(sym):
  oldsym = sym
  sym = defaultpg[oldsym]
  if verbose:
   print "                                                                   "
   print "************************** W A R N I N G **************************"
   print "Warning! You have chosen a crystal class (", oldsym, ") with more  "
   print "than one independent form of the elastic tensor! I am defaulting to"
   print "point group", defaultpg[oldsym], ".                                "
   print "************************** W A R N I N G **************************"
   print "                                                                   "
# Initialize projector
 projector=np.zeros((21,21))
# Obtain matrix elements
# Isotropic
 if sym == "iso":
  c1 = 1./5. ; c2 = np.sqrt(2.)/15. ; c3 = 2./15. ; c4 = 4./15.
  projector[0][0] = c1   ; projector[0][1] = c2   ; projector[0][2] = c2
  projector[0][6] = c1   ; projector[0][7] = c2   ; projector[0][11] = c1
  projector[0][15] = c3  ; projector[0][18] = c3  ; projector[0][20] = c3
  projector[1][0] = c2   ; projector[1][1] = c4   ; projector[1][2] = c4
  projector[1][6] = c2   ; projector[1][7] = c4   ; projector[1][11] = c2
  projector[1][15] = -c2 ; projector[1][18] = -c2 ; projector[1][20] = -c2
  projector[2][0] = c2   ; projector[2][1] = c4   ; projector[2][2] = c4
  projector[2][6] = c2   ; projector[2][7] = c4   ; projector[2][11] = c2
  projector[2][15] = -c2 ; projector[2][18] = -c2 ; projector[2][20] = -c2
  projector[6][0] = c1   ; projector[6][1] = c2   ; projector[6][2] = c2
  projector[6][6] = c1   ; projector[6][7] = c2   ; projector[6][11] = c1
  projector[6][15] = c3  ; projector[6][18] = c3  ; projector[6][20] = c3
  projector[7][0] = c2   ; projector[7][1] = c4   ; projector[7][2] = c4
  projector[7][6] = c2   ; projector[7][7] = c4   ; projector[7][11] = c2
  projector[7][15] = -c2 ; projector[7][18] = -c2 ; projector[7][20] = -c2
  projector[11][0] = c1  ; projector[11][1] = c2  ; projector[11][2] = c2
  projector[11][6] = c1  ; projector[11][7] = c2  ; projector[11][11] = c1
  projector[11][15] = c3 ; projector[11][18] = c3 ; projector[11][20] = c3
  projector[15][0] = c3  ; projector[15][1] = -c2 ; projector[15][2] = -c2
  projector[15][6] = c3  ; projector[15][7] = -c2 ; projector[15][11] = c3
  projector[15][15] = c1 ; projector[15][18] = c1 ; projector[15][20] = c1
  projector[18][0] = c3  ; projector[18][1] = -c2 ; projector[18][2] = -c2
  projector[18][6] = c3  ; projector[18][7] = -c2 ; projector[18][11] = c3
  projector[18][15] = c1 ; projector[18][18] = c1 ; projector[18][20] = c1
  projector[20][0] = c3  ; projector[20][1] = -c2 ; projector[20][2] = -c2
  projector[20][6] = c3  ; projector[20][7] = -c2 ; projector[20][11] = c3
  projector[20][15] = c1 ; projector[20][18] = c1 ; projector[20][20] = c1
# Cubic
 if sym == "cub" or sym == "23" or sym == "m-3" or sym == "432" or sym == "-43m" or sym == "m-3m":
  c1 = 1./3.
  projector[0][0] = c1   ; projector[0][6] = c1   ; projector[0][11] = c1
  projector[1][1] = c1   ; projector[1][2] = c1   ; projector[1][7] = c1
  projector[2][1] = c1   ; projector[2][2] = c1   ; projector[2][7] = c1
  projector[6][0] = c1   ; projector[6][6] = c1   ; projector[6][11] = c1
  projector[7][1] = c1   ; projector[7][2] = c1   ; projector[7][7] = c1
  projector[11][0] = c1  ; projector[11][6] = c1  ; projector[11][11] = c1
  projector[15][15] = c1 ; projector[15][18] = c1 ; projector[15][20] = c1
  projector[18][15] = c1 ; projector[18][18] = c1 ; projector[18][20] = c1
  projector[20][15] = c1 ; projector[20][18] = c1 ; projector[20][20] = c1
# Hexagonal
 if sym == "hex" or sym == "6" or sym == "-6" or sym == "6/m" or sym == "622" \
    or sym == "6mm" or sym == "-62m" or sym == "6/mmm":
  c1 = 3./8. ; c2 = 1./4./np.sqrt(2.) ; c3 = 1./4. ; c4 = 3./4. ; c5 = -1./2./np.sqrt(2.)
  c6 = 1./2. ; c7 = 1.
  projector[0][0] = c1   ; projector[0][1] = c2   ; projector[0][6] = c1
  projector[0][20] = c3  ; projector[1][0] = c2   ; projector[1][1] = c4
  projector[1][6] = c2   ; projector[1][20] = c5  ; projector[2][2] = c6
  projector[2][7] = c6   ; projector[6][0] = c1   ; projector[6][1] = c2
  projector[6][6] = c1   ; projector[6][20] = c3  ; projector[7][2] = c6
  projector[7][7] = c6   ; projector[11][11] = c7 ; projector[15][15] = c6
  projector[15][18] = c6 ; projector[18][15] = c6 ; projector[18][18] = c6
  projector[20][0] = c3  ; projector[20][1] = c5  ; projector[20][6] = c3
  projector[20][20] = c6
# Trigonal (point groups 3 and -3)
 if sym == "3" or sym == "-3":
  c1 = 3./8. ; c2 = 1./4. ; c3 = 1./2. ; c4 = 3./4. ; c5 = 1./4./np.sqrt(2.)
  c6 = 1./2./np.sqrt(2.) ; c7 = 1.
  projector[0][0] = c1   ; projector[0][1] = c5   ; projector[0][6] = c1
  projector[0][20] = c2  ; projector[1][0] = c5   ; projector[1][1] = c4
  projector[1][6] = c5   ; projector[1][20] = -c6 ; projector[2][2] = c3
  projector[2][7] = c3   ; projector[3][3] = c2   ; projector[3][8] = -c2
  projector[3][19] = c6  ; projector[4][4] = c2   ; projector[4][9] = -c2
  projector[4][17] = -c6 ; projector[6][0] = c1   ; projector[6][1] = c5
  projector[6][6] = c1   ; projector[6][20] = c2  ; projector[7][2] = c3
  projector[7][7] = c3   ; projector[8][3] = -c2  ; projector[8][8] = c2
  projector[8][19] = -c6 ; projector[9][4] = -c2  ; projector[9][9] = c2
  projector[9][17] = c6  ; projector[11][11] = c7 ; projector[15][15] = c3
  projector[15][18] = c3 ; projector[17][4] = -c6 ; projector[17][9] = c6
  projector[17][17] = c3 ; projector[18][15] = c3 ; projector[18][18] = c3
  projector[19][3] = c6  ; projector[19][8] = -c6 ; projector[19][19] = c3
  projector[20][0] = c2  ; projector[20][1] = -c6 ; projector[20][6] = c2
  projector[20][20] = c3 
# Trigonal (point groups 32, 3m and -3m)
 if sym == "32" or sym == "3m" or sym == "-3m":
  c1 = 3./8. ; c2 = 1./4./np.sqrt(2.) ; c3 = 1./4. ; c4 = 3./4. ; c5 = 1./2.
  c6 = 1./2./np.sqrt(2.) ; c7 = 1.
  projector[0][0] = c1 ; projector[0][1] = c2 ; projector[0][6] = c1
  projector[0][20] = c3 ; projector[1][0] = c2 ; projector[1][1] = c4
  projector[1][6] = c2 ; projector[1][20] = -c6 ; projector[2][2] = c5
  projector[2][7] = c5 ; projector[3][3] = c3 ; projector[3][8] = -c3
  projector[3][19] = c6 ; projector[6][0] = c1 ; projector[6][1] = c2
  projector[6][6] = c1 ; projector[6][20] = c3 ; projector[7][2] = c5
  projector[7][7] = c5 ; projector[8][3] = -c3 ; projector[8][8] = c3
  projector[8][19] = -c6 ; projector[11][11] = c7 ; projector[15][15] = c5
  projector[15][18] = c5 ; projector[18][15] = c5 ; projector[18][18] = c5
  projector[19][3] = c6 ; projector[19][8] = -c6 ; projector[19][19] = c5
  projector[20][0] = c3 ; projector[20][1] = -c6 ; projector[20][6] = c3
  projector[20][20] = c5
# Tetragonal (point groups 4, -4, 4/m)
 if sym == "4" or sym == "-4" or sym == "4/m":
  c1 = 1./2. ; c2 = 1.
  projector[0][0] = c1   ; projector[0][6] = c1   ; projector[1][1] = c2
  projector[2][2] = c1   ; projector[2][7] = c1   ; projector[5][5] = c1
  projector[5][10] = -c1 ; projector[6][0] = c1   ; projector[6][6] = c1
  projector[7][2] = c1   ; projector[7][7] = c1   ; projector[10][5] = -c1
  projector[10][10] = c1 ; projector[11][11] = c2 ; projector[15][15] = c1
  projector[15][18] = c1 ; projector[18][15] = c1 ; projector[18][18] = c1
  projector[20][20] = c2
# Tetragonal (point groups 422, 4mm, -42m, 4/mmm)
 if sym == "422" or sym == "4mm" or sym == "-42m" or sym == "4/mmm":
  c1 = 1./2. ; c2 = 1.
  projector[0][0] = c1   ; projector[0][6] = c1   ; projector[1][1] = c2
  projector[2][2] = c1   ; projector[2][7] = c1   ; projector[6][0] = c1
  projector[6][6] = c1   ; projector[7][2] = c1   ; projector[7][7] = c1
  projector[11][11] = c2 ; projector[15][15] = c1 ; projector[15][18] = c1
  projector[18][15] = c1 ; projector[18][18] = c1 ; projector[20][20] = c2
# Orthorhombic
 if sym == "ort" or sym == "222" or sym == "mm2" or sym == "mmm":
  c1 = 1.
  projector[0][0] = c1   ; projector[1][1] = c1   ; projector[2][2] = c1
  projector[6][6] = c1   ; projector[7][7] = c1   ; projector[11][11] = c1
  projector[15][15] = c1 ; projector[18][18] = c1 ; projector[20][20] = c1
# Monoclinic
 if sym == "mon" or sym == "2" or sym == "2/m" or sym == "m" or sym == "-2":
  c1 = 1.
  projector[0][0] = c1   ; projector[1][1] = c1   ; projector[2][2] = c1
  projector[4][4] = c1   ; projector[6][6] = c1   ; projector[7][7] = c1
  projector[9][9] = c1   ; projector[11][11] = c1 ; projector[13][13] = c1
  projector[15][15] = c1 ; projector[17][17] = c1 ; projector[18][18] = c1
  projector[20][20] = c1
# Triclinic
 if sym == "tic" or sym == "1" or sym == "-1":
  c1 = 1.
  for i in range(0,21):
   projector[i][i] = c1
# Carry out the projection
 proj=np.dot(projector,vector_c_voigt)
 return proj
##################################################################################
# Creates the function to be minimized for an input elastic tensor
# given in Voigt notation, in terms of the rotation angles
def res_ela(t, c_voigt, sym = None, verbose = False):
 tx=t[0] ; ty=t[1] ; tz=t[2]
 c_cart=ela_voigt_to_cartesian(c_voigt)
 rot_c=rotate_ela(c_cart,[tx,ty,tz])
 rot_c_voigt=ela_cartesian_to_voigt(rot_c)
 rot_vector=vectorize_ela_voigt(rot_c_voigt)
 proj_rot_vector=project_ela(rot_vector, sym = sym, verbose = verbose)
 res=rot_vector-proj_rot_vector
 result=np.dot(res,res)
 return result
##################################################################################
# Checks for the possible symmetry projections and gives the Euclidean
# distance for each of them. This allows to find out the most probable underlying
# symmetry of the tensor. Note that the distance will in general be reduced
# as the number of independent elastic constants is allowed to increase (e.g. for
# a triclinic projection the distance is zero). The function accepts two modes:
# with and without rotation optimization. Setting verbose = True will print
# the info from the minimization routine. The list of symmetries to check is
# complete by default. The user can override this if they're only interested
# in a reduced set. This function requires Scipy.
def ela_dist(c_voigt, 
             symlist = ["iso", "cub", "hex", "3", "32", "4", "4mm", "ort", "mon"],
             rotate = False, xtol = 1e-8, verbose = True, printmin = False):
 from scipy.optimize import fmin
 disp = 0
 if printmin:
  disp = 1
 result = []
 if not rotate:
  print "                                                                   "
  print "************************** R E S U L T S **************************"
  print "Results without rotation optimization                              "
  print "                                                                   "
  print "Symmetry     Euclidean distance                                    "
  print "--------     ------------------                                    "
  for sym in symlist:
   v = vectorize_ela_voigt(c_voigt)
   vp = project_ela(v, sym)
   edist2 = np.dot(v-vp,v-vp)
   edist = np.sqrt(edist2)
   print "%8s            %7.2f GPa" % (sym, edist)
   result.append([sym, edist])
  print "************************** R E S U L T S **************************"
  print "                                                                   "
 if rotate:
  print "                                                                   "
  print "************************** R E S U L T S **************************"
  print "Results with rotation optimization                                 "
  print "                                                                   "
  print "Symmetry     Euclidean distance     Angles tx,     ty,     tz      "
  print "--------     ------------------     -------------------------------"
  for sym in symlist:
   topt = [0., 0., 0.]
   if sym != "iso":
    topt = fmin(res_ela, x0=[0,0,0], xtol=xtol, args=(c_voigt, sym, verbose), disp=disp)
   ct = ela_voigt_to_cartesian(c_voigt)
   rotct = rotate_ela(ct, topt)
   rot_voigt = ela_cartesian_to_voigt(rotct)
   v = vectorize_ela_voigt(rot_voigt)
   vp = project_ela(v, sym = sym, verbose=False)
   edist2 = np.dot(v-vp,v-vp)
   edist = np.sqrt(edist2)
   printangles = ["%7.2f" % topt[0], "%7.2f" % topt[1], "%7.2f" % topt[2]]
#  Take symmetry planes into account
   if sym == "iso":
    printangles = ["    n/a", "    n/a", "    n/a"]
   if sym == "hex" or sym == "6" or sym == "-6" or sym == "6/m" or sym == "622" \
      or sym == "6mm" or sym == "-62m" or sym == "6/mmm" or sym == "3" or sym == "-3" \
      or sym == "32" or sym == "3m" or sym == "-3m":
    printangles = ["%7.2f" % topt[0], "%7.2f" % topt[1], "    n/a"]
   print "%8s            %7.2f GPa       %s %s %s  deg." \
         % (sym, edist, printangles[0], printangles[1], printangles[2])
   result.append([sym, edist, topt[0], topt[1], topt[2]])
  print "************************** R E S U L T S **************************"
  print "                                                                   "
 return result
##################################################################################
##################################################################################
############### End of functions for stiffness tensor manipulation ###############
##################################################################################
##################################################################################
