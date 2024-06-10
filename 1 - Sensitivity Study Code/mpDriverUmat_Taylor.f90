program mpDriver
  ! 6 May 2023
  !
  ! This code runs material points in uniaxial tension
  ! or compression. It does not work with any shear
  ! stresses.
  !
  ! This code is being used for Objective 1 of JPR
  ! dissertation.
  !
  implicit none

  integer i, Ninc, j

  integer ntens,nstatev, nprops, ndir, nshr
  integer kinc, kspt, kstep, layer, noel, npt

  real*8 coords(3), dfgrd0(3,3), dfgrd1(3,3)
  real*8 dpred(1), drot(3,3), predef(1), time(2)
  real*8 celent, dtime, pnewdt, dtemp, rpl
  real*8 scd, spd, sse, temp
  real*8 dtime_initial
  ! dummy values that may have different dimensions
  ! but that probably wont get used
  real*8 drpldt

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
  real*8 totalStrainLoad, totalStrainUnLoad
  real*8 totalStrainX, totalStrainY, totalStrainZ
  real*8 fMax(3,3), Emat(3,3), df(3,3)

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
  character(len=255) :: pwd

  ! number of elements, counters, and number
  ! of steps
  integer mpnum, m, k, t, numsteps, a, numsims
  integer eulercount, simcount, Beginning_Simnum
  integer End_Simnum, fileoffset, kinc_counter
  integer straincount

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
  real*8, dimension (:), allocatable:: strainState
  real*8, dimension (:,:), allocatable :: EULER
  real*8, dimension (:,:), allocatable :: totalStrain

  ! varibles to calculate tangent stiffness
  real*8 strain0(6), stress0(6), ddsddeFD(6,6)
  real*8  stressSave0(6,6), stressSave(6,6)

  real*8, dimension (:,:), allocatable :: statevSave0
  real*8, dimension (:,:), allocatable :: statevSave

  logical fdtan
  !--------------------------------------------------------------------
  ! coords,     !  coordinates of Gauss pt. being evaluated
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
  ! time	! Step Time (1) and Total Time (2)
  ! celent      ! Characteristic element length
  ! drpldt      ! Variation of RPL w.r.t temp.
  ! dtemp       ! Increment of temperature
  ! dtime       ! Increment of time
  ! kinc        !Increment number
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
  !--------------------------------------------------------------------
  ! Integer Inputs

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
  mpnum = 27
  ! fileoffset is the number of files which are read in
  fileoffset = 4
  ! For cyclic loading need to have at least two steps
  ! Numseteps is 2*Number of cycles (e.g. 4 steps = 2 cycles)
  numsteps = 8
  !--------------------------------------------------------------------
  call getcwd(pwd)

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

  ! variables used in code but not in umat
  allocate (strainState(ntens))
  !--------------------------------------------------------------------
  open(3,file="Beginning_and_Ending_Simnum.inc")
  read(3,*)Beginning_Simnum,End_Simnum
  numsims = End_Simnum-Beginning_Simnum+1
  !print *, numsims

  ! Get texture from file
  open(2,file="texture.inc")
  do eulercount = 1,mpnum
     read(2,*)EULER1,EULER2,EULER3
     EULER(eulercount,1)  = EULER1
     EULER(eulercount,2)  = EULER2
     EULER(eulercount,3)  = EULER3
  enddo
  !--------------------------------------------------------------------
  ! print *, EULER

  ! Start loop which reads properties and opens
  ! file location to store results for each
  ! simulation
  do simcount = Beginning_Simnum,End_Simnum
     !write(*,*)'Simulation_Number = ',simcount

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
     !-----------------------------------------------------------------
     do m = 1,mpnum
        !write(*,*)'Material Point Number = ',m

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
        open(unit=m+fileoffset,file=trim(pwd)//"/"//&
             trim(adjustl(simcountchar))//"/"//&
             trim(adjustl(filenum))//".dat",recl=204)

        ! Assign texture
        props(25) = EULER(m,1)
        props(26) = EULER(m,2)
        props(27) = EULER(m,3)

        !write(*,*)'Euler Angles = ',EULER(m,:)

        cmname = "Material-1"
        !--------------------------------------------------------------
        ! For elastic range  uniaxial tensile test:
        ! Stress Ratio: S1:S2:S3=1:0:0
        ! Strain Ratio: E1:E2:E3=1:-0.5:-0.5
        totalStrainLoad    =  0.06D0
        totalStrainUnLoad  =  0.D0

        totalStrainX       =  totalStrainLoad
        totalStrainY       = -totalStrainLoad/2
        totalStrainZ       = -totalStrainLoad/2

        ! initial time increment. Can change within the strain 
        ! increment loop
        dtime_initial = 0.0125D0
        dtime = dtime_initial
        !--------------------------------------------------------------
        ! Number of increments in each Step
        ! Setting it equal to inverse of dtime since that makes
        ! it easier to deal with for Objective 1 in thesis
        Ninc = 1/dtime

        coords(1:ndir) = (/0.0D0,0.0D0,0.0D0  /)
        !strainState(1:ntens) = (/1.D0, -0.0D0, -0.0D0, 0.D0, 0.D0, 0.D0/)
        
        dstrain(1:3) = (/totalStrainX*dtime, totalStrainY*dtime,&
             totalStrainZ*dtime /)
        dstrain(4:6) = (/0, 0, 0 /)

        dstran(1:3) = (/log(1.D0+totalStrainX*dtime),&
             log(1.D0+totalStrainY*dtime),&
             log(1.D0+totalStrainZ*dtime) /)
        dstran(4:6) = (/0, 0, 0 /)
        !--------------------------------------------------------------
        ! initalize varibles
        kinc = 1
        kspt = 1
        kstep = 1
        layer = 1
        !noel = 1
        noel = m
        npt = 1

        !intialize variables
        !startTemp = 277.D0
        startTemp = 0.D0
        temp = startTemp
        stress(1:ntens) = (/0.D0, 0.D0, 0.D0, 0.D0, 0.D0, 0.D0/)
        do i = 1, nstatev
           statev(i) =  0.D0
           beginning_step_statev(i) = 0.D0
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

        dfgrd1 = dfgrd0
        !--------------------------------------------------------------
        do t = 1,numsteps
           ! if the time increment change for the previous step, reset
           ! it for the new step
           time(1) = 0.D0
           beginning_step_time(1) = 0.D0
           dtime = dtime_initial
           kinc_counter = 1
           kstep = t
           ! do while time.le.t (time <= t) instead of using Ninc
           ! Each step will be 1 second, therefore the loop will go
           ! until a whole number is reached for the time step
           !do i = 1, Ninc
           do while (time(2).le.dble(t))
           !do while ((strain(1).le.totalStrainLoad).and.&
           !     (strain(1).ge.totalStrainUnLoad))
              kinc = kinc_counter

              dfgrd0(1,1:3) = (/strain(1) + 1.D0, 0.0D0, 0.D0/)
              dfgrd0(2,1:3) = (/0.D0, strain(2) + 1.D0, 0.D0/)
              dfgrd0(3,1:3) = (/0.D0, 0.D0, strain(3) + 1.D0/)

              if (MOD(t,2).eq.1) then
                 ! if 1 it is an odd step which loads the material point
                 ! in tension
                 dstrain(1)= totalStrainX*dtime
                 dstrain(2)= totalStrainY*dtime
                 dstrain(3)= totalStrainZ*dtime
                 dstrain(4)= 0.d0
                 dstrain(5)= 0.d0
                 dstrain(6)= 0.d0
                 
                 strain(1:ntens) = strain(1:ntens) + dstrain(1:ntens)

                 if(strain(1).ge.totalStrainLoad) exit
              else
                 ! if 0 it is an even step which unloads the material point
                 ! back to a 0 strain
                 dstrain(1)= totalStrainX*dtime
                 dstrain(2)= totalStrainY*dtime
                 dstrain(3)= totalStrainZ*dtime
                 dstrain(4)= 0.d0
                 dstrain(5)= 0.d0
                 dstrain(6)= 0.d0

                 strain(1:ntens) = strain(1:ntens) - dstrain(1:ntens)

                 if(strain(1).le.totalStrainUnLoad) exit
              endif

              dfgrd1(1,1:3) = (/strain(1) + 1.D0, 0.0D0, 0.D0/)
              dfgrd1(2,1:3) = (/0.D0, strain(2) + 1.D0, 0.D0/)
              dfgrd1(3,1:3) = (/0.D0, 0.D0, strain(3) + 1.D0/)

              ! To get logarithmic strain ln(1+strain)
              ! this is saved as stran. Makes it easier for post processing
              stran(1)= log(dfgrd1(1,1))
              stran(2)= log(dfgrd1(2,2))
              stran(3)= log(dfgrd1(3,3))
              stran(4)= 0.d0
              stran(5)= 0.d0
              stran(6)= 0.d0

              dstran(1) = log(dfgrd1(1,1))-log(dfgrd0(1,1))
              dstran(2) = log(dfgrd1(2,2))-log(dfgrd0(2,2))
              dstran(3) = log(dfgrd1(3,3))-log(dfgrd0(3,3))
              dstran(4) = 0.D0
              dstran(5) = 0.D0
              dstran(6) = 0.D0

              call umat( &
                   stress,  statev,  ddsdde,  sse,     spd, &
                   scd,     rpl,     ddsddt,  drplde,  drpldt, &
                   stran,   dstran,  time,    dtime,   temp, &
                   dtemp,   predef,  dpred,   cmname,  ndir, &
                   nshr,    ntens,   nstatev, props,   nprops, &
                   coords,  drot,    pnewdt,  celent,  dfgrd0, &
                   dfgrd1,  noel,    npt,     layer,   kspt, &
                   kstep,   kinc )

              if (pnewdt.ge.1.0)then
                 ! can proceed to next increment
                 time(1)  = time(1) + dtime
                 time(2)  = time(2) + dtime
                 kinc_counter = kinc_counter + 1
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

                 ! Increase time step to make simulation run
                 ! faster up to whatever the initial time step
                 ! was set to
                 pnewdt = 1.1D0
                 ! New time increment
                 dtime = dtime*pnewdt
                 if(dtime.gt.dtime_initial)then
                    dtime = dtime_initial
                    pnewdt = 1.D0
                 endif

                 beginning_step_celent = celent

                 ! no stretch no shear
                 beginning_step_dfgrd0(1,1:3) = dfgrd0(1,1:3)
                 beginning_step_dfgrd0(2,1:3) = dfgrd0(2,1:3)
                 beginning_step_dfgrd0(3,1:3) = dfgrd0(3,1:3)

                 ! pnewdt is usually 1.0, but if an internal error in the Umat
                 ! is tripped, it often reduces the timestep
                 write(m+fileoffset,"(E,$)") time(2), stress(1)
                ! write(m+fileoffset,"(E,$)") stran(1), stran(2), stran(3)
                ! write(m+fileoffset,"(E,$)") stran(4), stran(5), stran(6)
                ! write(m+fileoffset,"(E,$)") stress(1), stress(2), stress(3)
                ! write(m+fileoffset,"(E,$)") stress(4), stress(5), stress(6)
                ! write(m+fileoffset,"(E,$)") statev(1), statev(177), statev(39)
                ! write(m+fileoffset,"(E,$)") strain(1), time(1)
                 write(m+fileoffset,"(E)"  )  
                 !write(*,*)'pnewdt=', pnewdt
                 !print *, strain
                 !print *, stress
                 !print *, "statev(79)", statev(79)
                 !print *, 'time=',time
                 !dtime = dtime_initial

              else
                 !write(*,*)'PNEWDT less than 1. Time step change', pnewdt
                 ! Return to time and strain at beginning of step (i.e. the
                 ! values at the end of the previous step)
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
                 ! New time increment
                 dtime = dtime*pnewdt
                 ! Reset pnewdt to 1
                 pnewdt = 1.D0
                 celent = beginning_step_celent

                 ! no stretch no shear
                 dfgrd0(1,1:3) = beginning_step_dfgrd0(1,1:3)
                 dfgrd0(2,1:3) = beginning_step_dfgrd0(2,1:3)
                 dfgrd0(3,1:3) = beginning_step_dfgrd0(3,1:3)
              endif

              if(dtime.lt.0.00000001)then
                 write(*,*)'dtime=',dtime
                 write(*,*)'pnewdt=',pnewdt
                 ! The stop command ends the whole program
                 stop 'TIME INCREMENT TOO SMALL - mpDriver'
              endif

           end do ! end strain increment loop  
        end do ! end of step
     end do ! end of material point loop
  end do ! end of numsim loop

end program mpDriver



