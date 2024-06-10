program mpDriver
! 13 April 2023
!
! This code takes in DFGRD1 values saved from an
! ABAQUS simulation which was run using inputs
! taken from Manchiraju and Anderson 2010 and 
! Yu 2013 to be used as a baseline simulation.
!
! The simulation was run using static direct
! forcing each increment to be the same.
! There were 10,000 increments per step
! The DFGRD1 values for each element were saved
! in files Element_#.txt and are imported into
! this code to be used.
!
! At the end of each step/beginning of the next
! ABAQUS outputs the same values. Need to delete
! one of the lines containing the same values 
! consecutively (if ABAQUS sim was run with 100 
! steps, then line 100+1 would be deleted and so on)
!

  implicit none

  integer i, Ninc, j, Inc_Initial, Inc_Total, d_inc
  integer I_stran, J_stran, ABAQUS_Inc_Num, ABAQUS_Inc_Count

  integer ntens,nstatev, nprops, ndir, nshr
  integer kinc, kspt, kstep, layer, noel, npt
  integer beginning_step_inc

  real*8 coords(3), dfgrd0(3,3), dfgrd1(3,3)
  real*8 dpred(1), drot(3,3), predef(1), time(2)
  real*8 celent, dtime, pnewdt, dtemp, rpl
  real*8 scd, spd, sse, temp, dfgrd1_T(3,3)
  real*8 dtime_initial, DFGRD_Increment
  real*8 GL_stran(3,3)
  ! dummy values that may have different dimensions
  ! but that probably wont get used
  real*8 drpldt, DELTAIJ, CE(3,3)

  real*8 beginning_step_temp
  real*8 beginning_step_sse
  real*8 beginning_step_spd
  real*8 beginning_step_scd
  real*8 beginning_step_rpl
  real*8 beginning_step_drpldt
  real*8 beginning_step_time(2)
  real*8 beginning_step_dtemp
  real*8 beginning_step_dpred(1)
  real*8 beginning_step_drot(3,3)
  real*8 beginning_step_celent
  real*8 beginning_step_dfgrd0(3,3)

  real*8 startTemp
  real*8 strainIncLoadX, strainIncLoadY, strainIncLoadZ
  real*8 strainIncLoadXY, strainIncLoadXZ, strainIncLoadYZ
  real*8 strainIncUnLoadX, strainIncUnLoadY, strainIncUnLoadZ
  real*8 strainIncUnLoadXY, strainIncUnLoadXZ, strainIncUnLoadYZ
  real*8 totalStrainLoadX, totalStrainLoadY, totalStrainLoadZ
  real*8 totalStrainLoadXY, totalStrainLoadXZ, totalStrainLoadYZ
  real*8 totalStrainUnLoadX, totalStrainUnLoadY, totalStrainUnLoadZ
  real*8 totalStrainUnLoadXY, totalStrainUnLoadXZ, totalStrainUnLoadYZ
  real*8 StrainRatioX, StrainRatioY, StrainRatioZ
  real*8 StrainRatioXY, StrainRatioXZ, StrainRatioYZ
  real*8 fMax(3,3), Emat(3,3), df(3,3)
  real*8 dummyvar

  character*80 cmname

  ! New variables to read in properties
  real*8 C11A,C12A,C44A,C11M,C12M,C44M,alpha_A,alpha_M
  real*8 f_c,gamma_0,Theta_T,Theta_Ref_Low,lambda_T
  real*8 gamma_dot_0,Am,S0,H0,SS,Ahard,QL,Theta_Ref_High
  real*8 EULER1,EULER2,EULER3,Bsat,b1,N_GEN_F,d_ir,dir
  real*8 mu
  real*8 C11A2,C12A2,C44A2
  ! For sequential file numbering with
  ! leading zeros in file name
  ! This is limited to 99,999 elements
  ! since the char length is 5
  ! increase if necessary, will also
  ! have to change the format specifier in
  ! the write command if this changes
  ! e.g. if it is 4, specifier becomes I5.4
  character(len=5) :: filenum
  character(len=5) :: simcountchar
  character(len=5) :: DFGRD_Count_Char
  character(len=255) :: pwd

  ! number of elements, counters, and number
  ! of steps
  integer mpnum, m, k, t, numsteps, a, numsims
  integer eulercount, simcount, Beginning_Simnum
  integer End_Simnum, fileoffset, straincount
  integer kinc_counter, DFGRD_Count

  real*8, dimension (:,:), allocatable :: ddsdde 
  real*8, dimension (:,:), allocatable :: beginning_step_ddsdde
  real*8, dimension (:),   allocatable :: ddsddt 
  real*8, dimension (:),   allocatable :: beginning_step_ddsddt
  real*8, dimension (:),   allocatable :: drplde
  real*8, dimension (:),   allocatable :: beginning_step_drplde
  real*8, dimension (:),   allocatable :: dstran
  real*8, dimension (:),   allocatable :: dstrain
  real*8, dimension (:),   allocatable :: dstrainLoad 
  real*8, dimension (:),   allocatable :: dstrainUnLoad
  real*8, dimension (:),   allocatable :: props 
  real*8, dimension (:),   allocatable :: statev 
  real*8, dimension (:),   allocatable :: beginning_step_statev
  real*8, dimension (:),   allocatable :: stran
  real*8, dimension (:),   allocatable :: beginning_step_stran
  real*8, dimension (:),   allocatable :: strain
  real*8, dimension (:),   allocatable :: beginning_step_strain
  real*8, dimension (:),   allocatable :: stress
  real*8, dimension (:),   allocatable :: beginning_step_stress

  ! variables used in code but not in umat
  real*8, dimension (:), allocatable :: strainState
  real*8, dimension (:,:), allocatable :: EULER
  real*8, dimension (:,:), allocatable :: DFGRD1_Load
  real*8, dimension (:,:), allocatable :: DFGRD1_UnLoad
  real*8, dimension (:,:,:), allocatable :: DFGRD1_Values

  ! varibles to calculate tangent stiffness
  real*8 strain0(6), stress0(6), ddsddeFD(6,6)
  real*8  stressSave0(6,6), stressSave(6,6)

  real*8, dimension (:,:), allocatable :: statevSave0
  real*8, dimension (:,:), allocatable :: statevSave

  ! coords,     ! coordinates of Gauss pt. being evaluated
  ! ddsdde,     ! Tangent Stiffness Matrix
  ! ddsddt,	! Change in stress per change in temperature
  ! dfgrd0,	! Deformation gradient at beginning of step
  ! dfgrd1,	! Deformation gradient at end of step
  ! dpred,	! Change in predefined state variables
  ! drplde,	! Change in heat generation per change in strain
  ! drot,	! Rotation matrix
  ! dstrain,	! Strain increment tensor stored in vector form
  ! predef,	! Predefined state vars dependent on field variables
  ! props,	! Material properties passed in
  ! statev,	! State Variables
  ! strain,	! Strain tensor stored in vector form
  ! stress,	! Cauchy stress tensor stored in vector form
  ! time	! Step Time and Total Time
  ! celent      ! Characteristic element length
  ! drpldt      ! Variation of RPL w.r.t temp.
  ! dtemp       ! Increment of temperature
  ! dtime       ! Increment of time
  ! kinc        ! Increment number
  ! kspt        ! Section point # in current layer
  ! kstep       ! Step number
  ! layer       ! layer number
  ! noel        ! element number
  ! npt         ! Integration point number
  ! pnewdt      ! Ratio of suggested new time increment/time increment
  ! rpl         ! Volumetric heat generation
  ! scd         ! “creep” dissipation
  ! spd         ! plastic dissipation
  ! sse         ! elastic strain energy
  ! temp        ! temperature

  ! Number of increments in each ABAQUS step
  ABAQUS_Inc_Num = 10000

  !      Integer Inputs
  ndir = 3
  nshr = 3
  ntens = ndir + nshr
  nstatev = 234
  nprops = 33
  ! Number of material points in simulation
  ! Make sure mpnum matches the number of rows in
  ! the .inc files. Name results files numerically 
  ! and have spreadsheet in Excel showing what 
  ! simulation corresponds to which proerties
  mpnum = 64
  ! fileoffset is the number of files which are read in
  fileoffset = 4
  ! For cyclic loading need to have at least two steps
  ! Numseteps is 2*Number of cycles (e.g. 4 steps = 2 cycles)
  numsteps = 2

  call getcwd(pwd)

  ! !     Dimension Reals
  ! !     Dimension Reals
  allocate (beginning_step_stress(ntens) )
  allocate (beginning_step_statev(nstatev) )
  allocate (beginning_step_drplde(ntens) )
  allocate (beginning_step_ddsddt(ntens) )
  allocate (beginning_step_ddsdde(ntens,ntens) )

  allocate (ddsdde(ntens,ntens) )
  allocate (ddsddt(ntens) )
  allocate (drplde(ntens) )
  allocate (dstran(ntens) )
  allocate (dstrain(ntens) )
  allocate (dstrainLoad(ntens) )
  allocate (dstrainUnLoad(ntens) )
  allocate (props(nprops) )
  allocate (statev(nstatev) )
  allocate (stran(ntens) )
  allocate (beginning_step_stran(ntens) )
  allocate (strain(ntens) )
  allocate (beginning_step_strain(ntens) )
  allocate (stress(ntens) )
  allocate (EULER(mpnum,3) )

  allocate (DFGRD1_Load(mpnum,9) )
  allocate (DFGRD1_UnLoad(mpnum,9) )
  allocate (DFGRD1_Values(ABAQUS_Inc_Num*numsteps,9,mpnum) )

  ! variables used in code but not in umat
  allocate (strainState(ntens))
  allocate (statevSave(nstatev,6) )
  allocate (statevSave0(nstatev,6) )

  ! In order to have this be as close to the
  ! ABAQUS simulations as possible, the DFGRD1
  ! values at the end of the increments in an ABAQUS 
  ! simulation are exported to a text file and
  ! read in here to be used.
  ! The files are saved as Element_#.txt in the 
  ! subdirectory /DFGRD1_Vals
  do DFGRD_Count = 1,mpnum
     ! Convert index DFGRD_Count  into file number
     ! string "(I5)" allocates a file name that can
     ! be 5 characters long. Can change to "(I5.5)"
     ! to put leading zeros in front of number
     ! if wanted. 
     ! e.g.: write(filenum,"(I5)") 1 --> 1
     ! e.g.: write(filenum,"(I5.5)") 1 --> 00001
     write(DFGRD_Count_Char,"(I5)") DFGRD_Count
     open(4,file=trim(pwd)//"/DFGRD1_Vals/Element_"//&
          trim(adjustl(DFGRD_Count_Char))//".txt")
     do ABAQUS_Inc_Count = 1,ABAQUS_Inc_Num*numsteps
        read(4,*) DFGRD1_Values(ABAQUS_Inc_Count,1,DFGRD_Count),&
             DFGRD1_Values(ABAQUS_Inc_Count,2,DFGRD_Count),&
             DFGRD1_Values(ABAQUS_Inc_Count,3,DFGRD_Count),&
             DFGRD1_Values(ABAQUS_Inc_Count,4,DFGRD_Count),&
             DFGRD1_Values(ABAQUS_Inc_Count,5,DFGRD_Count),&
             DFGRD1_Values(ABAQUS_Inc_Count,6,DFGRD_Count),&
             DFGRD1_Values(ABAQUS_Inc_Count,7,DFGRD_Count),&
             DFGRD1_Values(ABAQUS_Inc_Count,8,DFGRD_Count),&
             DFGRD1_Values(ABAQUS_Inc_Count,9,DFGRD_Count)
     enddo
  enddo
  
!  print *,  DFGRD1_Values(1,1,1),&
!       DFGRD1_Values(1,2,1),&
!       DFGRD1_Values(1,3,1),&
!       DFGRD1_Values(1,4,1),&
!       DFGRD1_Values(1,5,1),&
!       DFGRD1_Values(1,6,1),&
!       DFGRD1_Values(1,7,1),&
!       DFGRD1_Values(1,8,1),&
!       DFGRD1_Values(1,9,1)

  open(3,file="Beginning_and_Ending_Simnum.inc")
  read(3,*)Beginning_Simnum,End_Simnum
  numsims = End_Simnum-Beginning_Simnum+1
  !print *, numsims

  ! Get texture from file
  open(2,file="texture.inc")
  do eulercount = 1,mpnum
     read(2,*)EULER(eulercount,1),EULER(eulercount,2),EULER(eulercount,3)
  enddo

  ! print *, EULER

  ! Start loop which reads properties and opens
  ! file location to store results for each
  ! simulation
  do simcount = Beginning_Simnum,End_Simnum

     ! Euler angles and properties should be in a
     ! text file where each row contains the values
     ! for each material point/simulation
     open(1,file="Properties_mpDriver.inc")
     !open(1,file="Props.inc")

     ! Read in material properties
     ! Note the Fortran read() statement reads a 
     ! row of values and then advances to the
     ! beginning of the next line.
     read(1,*)C11A,C12A,C44A,C11M,C12M,C44M,alpha_A,alpha_M, &
          f_c,gamma_0,Theta_T,Theta_Ref_Low,lambda_T,C11A2,C12A2,C44A2, &
          gamma_dot_0,Am,S0,H0,SS,Ahard,QL,Theta_Ref_High, &
          Bsat,b1,N_GEN_F,d_ir,dir,mu

!     print *, C11A,C12A,C44A,C11M,C12M,C44M,alpha_A,alpha_M, &
!          f_c,gamma_0,Theta_T,Theta_Ref_Low,lambda_T,C11A2,C12A2,C44A2, &
!          gamma_dot_0,Am,S0,H0,SS,Ahard,QL,Theta_Ref_High, &
!          Bsat,b1,N_GEN_F,d_ir,dir,mu

     ! Assigning properties which were read in from
     ! the .inc file
     props(1)  = C11A
     props(2)  = C12A
     props(3)  = C44A
     props(4)  = C11M
     props(5)  = C12M
     props(6)  = C44M
     props(7)  = alpha_A
     props(8)  = alpha_M
     props(9)  = f_c
     props(10) = gamma_0
     props(11) = Theta_T
     props(12) = Theta_Ref_Low
     props(13) = lambda_T
     props(14) = C11A2
     props(15) = C12A2
     props(16) = C44A2
     props(17) = gamma_dot_0
     props(18) = Am
     props(19) = S0
     props(20) = H0
     props(21) = SS
     props(22) = Ahard
     props(23) = QL
     props(24) = Theta_Ref_High
     !props(25) = EULER1
     !props(26) = EULER2
     !props(27) = EULER3
     props(28) = Bsat
     props(29) = b1
     props(30) = N_GEN_F
     props(31) = d_ir
     props(32) = dir
     props(33) = mu 

     do m = 1,mpnum

        print *, 'Simulation Nnumber = ',simcount
        print *, 'Elelemt Number = ',m
        ! Convert index m into file number string
        ! "(I5)" allocates a file name that can be
        ! 5 characters long. Can change to "(I5.5)"
        ! to put leading zeros in front of number
        ! if wanted. 
        ! e.g.: write(filenum,"(I5)") 1 --> 1
        ! e.g.: write(filenum,"(I5.5)") 1 --> 00001
        write(filenum,"(I5)") m
        write(simcountchar,"(I5)") simcount

        ! Convert index simcount to string

        ! Uncomment if using external file for texture
        !read(2,*)EULER1,EULER2,EULER3

        ! Open a file to allocate where the results
        ! will be stored. In this code, it will make
        ! a file for each material point (mpnum)
        ! Note: // joins two strings together
        open(unit=m+fileoffset,file=trim(pwd)//"/"&
             //trim(adjustl(simcountchar))//"/"//&
             trim(adjustl(filenum))//".dat",recl=204)

        ! Assign texture
        props(25) = EULER(m,1)
        props(26) = EULER(m,2)
        props(27) = EULER(m,3)

        cmname = "Material-1"

        ! For the ABAQUS_Inc_Num step ABAQUS simulation that was
        ! run, it means that the mpDriver number of steps
        ! will be ABAQUS_Inc_Num/Inc_Initial and if there is a 
        ! request from the Umat to change the time (i.e., PNEWDT
        ! is not 1), then it will decrease by a factor of 5 until
        ! it reaches a value of 1. If it tries to go below 1, the
        ! program will end
        Inc_Initial = 125
        d_inc = Inc_Initial
        dtime_initial = (1.D0/(ABAQUS_Inc_Num/Inc_Initial))
        dtime = dtime_initial

        ! Number of increments in each Step
        ! Setting it equal to inverse of dtime since that makes
        ! it easier to deal with for Objective 1 in thesis
        Ninc = 1/dtime
        coords(1:ndir) = (/0.0D0,0.0D0,0.0D0 /)

        ! initalize varibles
        kinc = 1
        kspt = 1
        kstep = 1
        layer = 1
        ! noel = 1
        noel = m
        npt = 1

        !intialize variables
        startTemp = 0.D0
        temp = startTemp
        stress(1:ntens) = (/0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0/)
        do i = 1, nstatev
           statev(i) =  0.D0
           beginning_step_statev(i) =  0.D0
        enddo
        ddsdde(:,:) = 0.D0
        sse = 0.D0
        spd = 0.D0
        scd = 0.D0
        rpl = 0.D0
        ddsddt(1:ntens) = (/0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0/)
        drplde(1:ntens) = (/0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0/)
        drpldt = 0.D0
        strain(1:ntens) = (/0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0/)
        stran(1:ntens) = (/0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0/)
        time(1) = 0.D0
        time(2) = 0.D0
        dtemp = 0.D0
        dpred = 0.D0
        drot(:,:) = 0.D0
        pnewdt = 1.D0
        celent = 1.D0
        
        beginning_step_strain(1:ntens) = (/0.D0, 0.D0, 0.D0,&
             0.D0, 0.D0, 0.D0/)
        beginning_step_time(1) = 0.D0
        beginning_step_time(2) = 0.D0

        ! no stretch no shear
        dfgrd0(1,1:3) = (/1.D0, 0.D0, 0.D0/)
        dfgrd0(2,1:3) = (/0.D0, 1.D0, 0.D0/)
        dfgrd0(3,1:3) = (/0.D0, 0.D0, 1.D0/)

        beginning_step_dfgrd0 = dfgrd0

        dfgrd1 = dfgrd0
        t = 1
        Inc_Total = Inc_Initial

        do while (Inc_Total.le.ABAQUS_Inc_Num*numsteps)
           !print *, 'Inc_Total=',Inc_Total
           !print *, 'statev(1)=',statev(1)
           !kinc = kinc_counter

           dfgrd0(1,1) = beginning_step_dfgrd0(1,1)
           dfgrd0(1,2) = beginning_step_dfgrd0(1,2)
           dfgrd0(1,3) = beginning_step_dfgrd0(1,3)
           dfgrd0(2,1) = beginning_step_dfgrd0(2,1)
           dfgrd0(2,2) = beginning_step_dfgrd0(2,2)
           dfgrd0(2,3) = beginning_step_dfgrd0(2,3)
           dfgrd0(3,1) = beginning_step_dfgrd0(3,1)
           dfgrd0(3,2) = beginning_step_dfgrd0(3,2)
           dfgrd0(3,3) = beginning_step_dfgrd0(3,3)
           
           dfgrd1(1,1) = DFGRD1_Values(Inc_Total,1,m)
           dfgrd1(1,2) = DFGRD1_Values(Inc_Total,2,m)
           dfgrd1(1,3) = DFGRD1_Values(Inc_Total,3,m)
           dfgrd1(2,1) = DFGRD1_Values(Inc_Total,4,m)
           dfgrd1(2,2) = DFGRD1_Values(Inc_Total,5,m)
           dfgrd1(2,3) = DFGRD1_Values(Inc_Total,6,m)
           dfgrd1(3,1) = DFGRD1_Values(Inc_Total,7,m)
           dfgrd1(3,2) = DFGRD1_Values(Inc_Total,8,m)
           dfgrd1(3,3) = DFGRD1_Values(Inc_Total,9,m)

           ! For Green-Lagrange strain calculation
           dfgrd1_T = transpose(dfgrd1)
           CE = matmul(dfgrd1_T,dfgrd1)

           ! Calculate overall strain
           do I_stran=1,3
              do J_stran=1,3
                 if(I_stran.eq.J_stran)then
                    DELTAIJ=1.0
                 else
                    DELTAIJ=0.0
                 end if
                 GL_stran(I_stran,J_stran)=0.50*(CE(I_stran,J_stran)-DELTAIJ)
              end do
           end do

           ! In Abaqus, the stresses and strains are arranged as follows
           ! 11, 22, 33, 12, 23, 31
           ! XX, YY, ZZ, XY, YZ, ZX
           strain(1) = GL_stran(1,1)
           strain(2) = GL_stran(2,2)
           strain(3) = GL_stran(3,3)
           strain(4) = GL_stran(1,2)
           strain(5) = GL_stran(2,3)
           strain(6) = GL_stran(3,1)

           dstrain(1) = strain(1) - beginning_step_strain(1)
           dstrain(2) = strain(2) - beginning_step_strain(2)
           dstrain(3) = strain(3) - beginning_step_strain(3)
           dstrain(4) = strain(4) - beginning_step_strain(4)
           dstrain(5) = strain(5) - beginning_step_strain(5)
           dstrain(6) = strain(6) - beginning_step_strain(6)

           ! Not sure if dstran is really needed since Umat does not
           ! use stran or dstran when called

           call umat( &
                stress,  statev,  ddsdde,  sse,     spd, &
                scd,     rpl,     ddsddt,  drplde,  drpldt, &
                strain,  dstrain, time,    dtime,   temp, &
                dtemp,   predef,  dpred,   cmname,  ndir, &
                nshr,    ntens,   nstatev,  props,   nprops, &
                coords,  drot,    pnewdt,  celent,  dfgrd0, &
                dfgrd1,  noel,    npt,     layer,   kspt, &
                kstep,   kinc )
           if(stress(1).ne.stress(1))then
              stop'NaN Stress Values'
           endif

           if (pnewdt.ge.1.0)then
              ! can proceed to next increment
              ! step at end of this increment
              beginning_step_inc = Inc_Total
              
              time(1)  = time(1) + dtime
              time(2)  = time(2) + dtime
              kinc = kinc + 1
              ! Save time and strain so that if the next step has a decrease 
              ! in time increment, the time and strain can be reset to 
              ! whatever the previous increment values were.
              beginning_step_temp = temp
              beginning_step_stress(1:ntens)=stress(1:ntens)
              do i = 1, nstatev
                 beginning_step_statev(i) =  statev(i)
              enddo
              do i = 1,ntens
                 do j = 1,ntens
                    beginning_step_ddsdde(i,j) = ddsdde(i,j)
                 enddo
              enddo
              beginning_step_sse = sse
              beginning_step_spd = spd
              beginning_step_scd = scd
              beginning_step_rpl = rpl
              beginning_step_ddsddt(1:ntens) = ddsddt(1:ntens)
              beginning_step_drplde(1:ntens) = drplde(1:ntens)
              beginning_step_drpldt = drpldt
              beginning_step_strain(1:ntens) = strain(1:ntens)
              beginning_step_stran(1:ntens) = stran(1:ntens)
              beginning_step_time(1) = time(1)
              beginning_step_time(2) = time(2)
              beginning_step_dtemp = dtemp
              beginning_step_dpred = dpred
              do i = 1,3
                 do j = 1,3
                    beginning_step_drot(i,j) = drot(i,j)
                 enddo
              enddo
              ! Reset pnewdt to 1
              pnewdt = 1.D0

              beginning_step_celent = celent

              ! DFGRD1 is DFGRD0 in the next step
              beginning_step_dfgrd0(1,1:3) = dfgrd1(1,1:3)
              beginning_step_dfgrd0(2,1:3) = dfgrd1(2,1:3)
              beginning_step_dfgrd0(3,1:3) = dfgrd1(3,1:3)

              !print *, 'Inc_Total=', Inc_Total

              ! pnewdt is usually 1.0, but if an internal error in the Umat
              ! is tripped, it often reduces the timestep
              write(m+fileoffset,"(E,$)") time(2)
              write(m+fileoffset,"(E,$)") strain(1), strain(2), strain(3)
              write(m+fileoffset,"(E,$)") strain(4), strain(5), strain(6)
              write(m+fileoffset,"(E,$)") stress(1), stress(2), stress(3)
              write(m+fileoffset,"(E,$)") stress(4), stress(5), stress(6)
              !write(m+fileoffset,"(E,$)") statev(1), statev(177), statev(39)
              !write(m+fileoffset,"(E,$)") dfgrd1(1,1),dfgrd1(2,2),dfgrd1(3,3)
              !write(m+fileoffset,"(E,$)") dtime
              !write(m+fileoffset,*) Inc_Total
              write(m+fileoffset,"(E)"  ) 
              !write(*,*)'pnewdt=', pnewdt
              !print *, strain
              !print *, stress
              !print *, "statev(79)", statev(79)
              !print *, 'time=',time
              !dtime = dtime_initial

              ! New value of step total for next increment
              Inc_Total = Inc_Total + d_inc

              if ((ABAQUS_Inc_Num/Inc_Total.gt.t).and.(Inc_Total.gt.0))then
                 t = t + 1
                 time(1) = 0.D0
                 beginning_step_time(1) = 0.D0
                 d_inc = Inc_Initial
                 dtime = dtime_initial
                 kinc = 1
                 kstep = t
              endif

           else
              write(*,*)'PNEWDT less than 1. Time step change', pnewdt
              ! Return to time and strain at beginning of step (i.e. the
              ! values at the end of the previous step)

              if ((Inc_Total.ge.Inc_Initial).and.(Inc_Total.ge.1))then
                  d_inc =  d_inc/5
                 Inc_Total = Inc_Total/5
              else
                 stop 'Requested Initial Increment too small'
              endif

              Inc_Total = beginning_step_inc + d_inc
              dtime = (1.D0/(ABAQUS_Inc_Num/d_inc))

              if (d_inc.lt.1)then
                 stop 'Requested Increment too small'
              endif

              temp = beginning_step_temp
              stress(1:ntens) = beginning_step_stress(1:ntens)
              do i = 1, nstatev
                 statev(i) =  beginning_step_statev(i)
              enddo
              do i = 1,ntens
                 do j = 1,ntens
                    ddsdde(i,j) = beginning_step_ddsdde(i,j)
                 enddo
              enddo
              sse = beginning_step_sse
              spd = beginning_step_spd
              scd = beginning_step_scd
              rpl = beginning_step_rpl
              ddsddt(1:ntens) = beginning_step_ddsddt(1:ntens)
              drplde(1:ntens) = beginning_step_drplde(1:ntens)
              drpldt = beginning_step_drpldt
              strain(1:ntens) = beginning_step_strain(1:ntens)
              stran(1:ntens) = beginning_step_stran(1:ntens)
              time(1) = beginning_step_time(1)
              time(2) = beginning_step_time(2)
              dtemp = beginning_step_dtemp
              dpred = beginning_step_dpred
              do i = 1,3
                 do j = 1,3
                    drot(i,j) = beginning_step_drot(i,j)
                 enddo
              enddo

              ! Reset pnewdt to 1
              pnewdt = 1.D0
              celent = beginning_step_celent

              dfgrd0(1,1:3) = beginning_step_dfgrd0(1,1:3)
              dfgrd0(2,1:3) = beginning_step_dfgrd0(2,1:3)
              dfgrd0(3,1:3) = beginning_step_dfgrd0(3,1:3)
           endif

        end do ! end strain increment loop  
     end do ! end of material point loop
  end do ! end of numsim loop

end program mpDriver
