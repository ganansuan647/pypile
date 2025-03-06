c
c**************************************************************
c    Sub to calculate areas at the bottom of piles
c**************************************************************
c
          SUBROUTINE AREA(PNUM,ZFR,ZBL,AO)
        COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     &    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     &    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     &    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
          INTEGER PNUM
          REAL ZFR(PNUM),ZBL(PNUM),AO(PNUM)
          DIMENSION BXY(1000,2),W(1000),SMIN(1000)
          DO 81 K=1,PNUM
            BXY(K,1)=PXY(K,1)+(ZFR(K)+ZBL(K))*AGL(K,1)
            BXY(K,2)=PXY(K,2)+(ZFR(K)+ZBL(K))*AGL(K,2)
            IF(KSU(K).GT.2) THEN
               IF(NBL(K).NE.0) W(K)=DOB(K,NBL(K))
               IF(NBL(K).EQ.0) W(K)=DOF(K,NFR(K))
              GOTO 81
            END IF
            W(K)=0.0
            AG=ATAN(SQRT(1-AGL(K,3)**2)/AGL(K,3))
C             AG1=AG*180.0/3.142
C             WRITE(*,'(4HAG= ,F10.4)') AG1
            DO 80 IA=1,NBL(K)
80            W(K)=W(K)+HBL(K,IA)*(SIN(AG)-AGL(K,3)*
     !             TAN(AG-PFI(K,IA)*3.142/720.0))
            W(K)=W(K)*2+DOB(K,1)
81          SMIN(K)=100.0
          DO 82 K=1,PNUM
            DO 82 IA=K+1,PNUM
              S=SQRT((BXY(K,1)-BXY(IA,1))**2+
     !           (BXY(K,2)-BXY(IA,2))**2)
              IF(S.LT.SMIN(K)) SMIN(K)=S
              IF(S.LT.SMIN(IA)) SMIN(IA)=S
82        CONTINUE
          DO 83 K=1,PNUM
            IF(SMIN(K).LT.W(K)) W(K)=SMIN(K)
            IF(KSH(K).EQ.0) AO(K)=3.142*W(K)**2/4.0
            IF(KSH(K).EQ.1) AO(K)=W(K)**2
83        CONTINUE
C          WRITE(*,'(/16HAreas at bottom:,5F10.4)') (AO(K),K=1,PNUM)
          RETURN
          END