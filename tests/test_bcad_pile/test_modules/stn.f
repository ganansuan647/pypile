c*****************************************************************
c    Sub to calculate axial stiffness of a single pile
c*****************************************************************
c
      SUBROUTINE STN(K,ZBL,AO,RZZ)
    COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
      IF(KSU(K).EQ.1) PKC=0.5
      IF(KSU(K).EQ.2) PKC=0.667
      IF(KSU(K).GT.2) PKC=1.0
      X=0.0
      DO 90 IA=1,NFR(K)
        CALL EAJ(KSH(K),PKE(K),DOF(K,IA),A,B)
90        X=X+HFR(K,IA)/(PEH(K)*A)
      DO 91 IA=1,NBL(K)
        CALL EAJ(KSH(K),PKE(K),DOB(K,IA),A,B)
91        X=X+PKC*HBL(K,IA)/(PEH(K)*A)
      IF(KSU(K).LE.2) X=X+1.0/(PMB(K)*ZBL*AO)
      IF(KSU(K).GT.2) X=X+1.0/(PMB(K)*AO)
      RZZ=1.0/X
      RETURN
      END
      
c**************************************************************
c     Sub to calculate properties of pile cross section
c**************************************************************
c
      SUBROUTINE EAJ(J,PKE,DO,A,B)
      IF(J.EQ.0) THEN
        A=3.142*DO**2/4.0
        B=PKE*3.142*DO**4/64.0
      ELSE
        A=DO**2
        B=PKE*DO**4/12.0
      END IF
      RETURN
      END
