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
#                               MattPy v0.1                                      #
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
###             Distribution last updated on 30 Sept. 2015                     ###
#####                                                                        #####
##################################################################################
##################################################################################



# Load dependencies (some functions might also require scipy, which is then
# loaded inside the function definition)
import numpy as np



##################################################################################
##### All the functions for manipulation of piezoelectric tensors are below  #####
##################################################################################
# Turns PZ tensor in Voigt notation to vector (preserving the norm)
# d_ij and e_ij forms have a different vector representation
def vectorize_pz_voigt(e_voigt, form = None, verbose = True):
 if not form or form not in ["e", "d"]:
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

# Turns PZ vector (assumed to preserve the norm) to tensor in Voigt notation
def tensorize_pz_voigt(vector_e_voigt, form = None):
 if not form or form not in ["e", "d"]:
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


# Transforms PZ tensor in Voigt notation to Cartesian notation
def pz_voigt_to_cartesian(e_voigt, form = None, verbose = True):
 if not form or form not in ["e", "d"]:
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


# Transforms PZ tensor in Cartesian notation to Voigt notation
def pz_cartesian_to_voigt(e_cart, form = None, verbose = True):
 if not form or form not in ["e", "d"]:
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


# Performs a rotation operation on a (Cartesian) rank-3 tensor
def rotate_pz(e_cart,rot_angles):
 result=np.zeros((3,3,3))
 tx=rot_angles[0] ; ty=rot_angles[1] ; tz=rot_angles[2]
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
 defaultpg = {"cub": "-43m", "hex": "6mm", "tig": "3m", "tet": "4mm", "ort" :"222", "mon": "2", "tric": "1"}
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
 result=np.dot(projector,vector_e_voigt)
 return result

# Creates the function to be minimized for an input PZ tensor
# given in Voigt notation, in terms of the rotation angles
def res_pz(t, e_voigt, sym = None, form = None, verbose = True):
 tx=t[0] ; ty=t[1] ; tz=t[2]
 e_cart=pz_voigt_to_cartesian(e_voigt, form = form)
 rot_e=rotate_pz(e_cart,[tx,ty,tz])
 rot_e_voigt=pz_cartesian_to_voigt(rot_e, form = form)
 rot_vector=vectorize_pz_voigt(rot_e_voigt, form = form, verbose = verbose)
 proj_rot_vector=project_pz(rot_vector,sym, verbose = verbose)
 res=rot_vector-proj_rot_vector
 result=np.dot(res,res)
 return result


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
   v = vectorize_pz_voigt(e_voigt, form = form, verbose = False)
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
   v = vectorize_pz_voigt(rot_voigt, form = form, verbose = False)
   vp = project_pz(v, sym = sym, verbose=False)
   edist2 = np.dot(v-vp,v-vp)
   edist = np.sqrt(edist2)
   topt[0] *= 180./np.pi ; topt[1] *= 180./np.pi ; topt[2] *= 180./np.pi
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
############# End of functions for piezoelectric tensor manipulation #############
##################################################################################




##################################################################################
####### All the functions for manipulation of stiffness tensors are below  #######
##################################################################################
# Turns elastic tensor in Voigt notation to vector (preserving the norm)
# it also symmetrizes the tensor in case it's not already symmetric
def vectorize_ela_voigt(c_voigt, verbose = True):
 result=[]
 flag = False
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
   if np.abs(c_voigt[i][j] - c_voigt[j][i]) > 0.01 and not flag and verbose:
    print "                                                                   "
    print "************************** W A R N I N G **************************"
    print "Warning! Your elastic tensor is not symmetric, I'm symmetrizing it!"
    print "************************** W A R N I N G **************************"
    print "                                                                   "
    flag = True
 return result


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


# Performs a rotation operation on a (Cartesian) rank-4 tensor
def rotate_ela(c_cart, rot_angles):
 result=np.zeros((3,3,3,3))
 tx=rot_angles[0] ; ty=rot_angles[1] ; tz=rot_angles[2]
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
 result=np.dot(projector,vector_c_voigt)
 return result


# Creates the function to be minimized for an input elastic tensor
# given in Voigt notation, in terms of the rotation angles
def res_ela(t, c_voigt, sym = None, verbose = False):
 tx=t[0] ; ty=t[1] ; tz=t[2]
 c_cart=ela_voigt_to_cartesian(c_voigt)
 rot_c=rotate_ela(c_cart,[tx,ty,tz])
 rot_c_voigt=ela_cartesian_to_voigt(rot_c)
 rot_vector=vectorize_ela_voigt(rot_c_voigt, verbose = verbose)
 proj_rot_vector=project_ela(rot_vector, sym = sym, verbose = verbose)
 res=rot_vector-proj_rot_vector
 result=np.dot(res,res)
 return result


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
   v = vectorize_ela_voigt(c_voigt, verbose)
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
   topt[0] *= 180./np.pi ; topt[1] *= 180./np.pi ; topt[2] *= 180./np.pi
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
############### End of functions for stiffness tensor manipulation ###############
##################################################################################
