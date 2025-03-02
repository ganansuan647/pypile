c
c                             BCAD_PILE
c
c    This program is aimed to execute spatial statical analysis of pile
c                  foundations of bridge substructures.
c
c
c***********************************************************************
c          Main routine of the BCAD_PILE program
c***********************************************************************
c
c     The function of this main routine is to control the procedure of 
c     the program execution.
c

c          SUBROUTINE BCAD_PILE(SO,FORCE,JCTR,INO)
c
c       /PINF/ is the name of a common area in which the information
c       of non_simulative piles is kept.
c

        character*20 fname
        character*24 file_name1,file_name2,file_name3



        COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
c
c       /SIMU/  is the name of a common area in which the information
c       of simulative piles is kept.
c
          COMMON /SIMU/ SXY(20,2),KSCTR(20)
c
c       /ESTIF/ is the name of a common area in which element stiffnesses
c       of each pile are kept. 
c
          COMMON /ESTIF/ ESP(1000000,6)
c
          INTEGER PNUM,SNUM
          DIMENSION FORCE(6),ZFR(1000),ZBL(1000),AO(1000),RZZ(1000),
     !              DUK(1000,6),BTX(1000,15),BTY(1000,15),SO(6,6)
c           To call a sub to creat program head
           CALL HEAD1
c           To call a sub to read input information from an input file

          WRITE(*,'(/7X,27HPlease enter data filename:/)')
          READ(*,'(A20)')fname
C          WRITE(*,'(34X,A15)')FNAME
           Call f_name(fname,'.dat',file_name1)
C           write(*,*) fname

          OPEN(51,FILE=file_NAME1)
           Call f_name(fname,'.out',file_name2)
           Call f_name(fname,'.pos',file_name3)
c           To open an output file    
          OPEN(52,FILE=file_name2,STATUS='UNKNOWN')
          OPEN(53,FILE=file_name3,STATUS='UNKNOWN')
c           To call a sub to creat program head
           CALL HEAD2

          WRITE(*,600)
          CALL R_DATA(JCTR,INO,PNUM,SNUM,FORCE,ZFR,ZBL)
c           To call a sub to calculate deformation factors of piles
          WRITE(*,610)
          CALL BTXY(PNUM,ZFR,ZBL,BTX,BTY)        
c           To call a sub to calculate areas at the bottom of piles
          WRITE(*,620)
          CALL AREA(PNUM,ZFR,ZBL,AO)
c           To call a sub to calculate axis stiffness of piles 
          CALL STIFF_N(PNUM,ZFR,ZBL,AO,RZZ)
c           To call a sub to calculate lateral stiffness of piles
          WRITE(*,630)
          CALL PSTIFF(PNUM,RZZ,BTX,BTY)
c           To call a sub to calculate displacements of the cap of the
c             pile foundation 
          WRITE(*,640)
          CALL DISP(JCTR,INO,PNUM,SNUM,PXY,SXY,AGL,FORCE,DUK,SO)
c           To call a sub to calculate displacements and internal forces
c               of each pile body
          CALL EFORCE(PNUM,BTX,BTY,ZBL,DUK)
c
600    FORMAT(7X,'*** To read input information ***'/)
610    FORMAT(//7X,'*** To calculate deformation factors of piles ***')
620    FORMAT(//7X,'*** To calculate axis stiffness of piles ***')
630    FORMAT(//7X,'*** To calculate lateral stiffness of piles ***')
640    FORMAT(//7X,'*** To execute entire pile foundation analysis ***
     !  '//)
          STOP         
          END

c*********************************************************************
c            Sub to read initial structural data 
c*********************************************************************
          SUBROUTINE R_DATA(JCTR,INO,PNUM,SNUM,FORCE,ZFR,ZBL)
C    
C     Declaration of arguments:
c         PNUM--      number of non-simulative piles
c         SNUM--      number of simulative piles
c         FORCE(6)--  external forces
c         ZFR(60)--   Length of each pile above ground
c         ZFR(60)--   Length of each pile beneath ground
c
          CHARACTER*4 TAG,SIG(30),CH*1,STAG*3
          CHARACTER*9 TITLE,NAME*15
C
C     Information of non-simulative piles is kept in a common area whose 
C       name is 'PINF'.
C
C
        COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
C
C     Information of simulative piles is kept in a common area whose
C       name is 'SIMU'.
C
          COMMON /SIMU/ SXY(20,2),KSCTR(20)
C          
          COMMON /ESTIF/ ESP(1000000,6)
C
          DIMENSION FORCE(6),ZFR(1000),ZBL(1000),JNEW(1000),VNEW(1000),
     !           AXY(10,2),ACT(10,6),nfb(1000),zfb(1000,1000)
          INTEGER PNUM,SNUM
c

c
c           To read input information of [CONTRAL] block
c
          READ(51,'(A9)') TITLE
          READ(51,*) JCTR
          IF(JCTR.EQ.1) THEN
            READ(51,*) NACT
            DO 10 I=1,NACT
            READ(51,*) (AXY(I,J),J=1,2)
            READ(51,*) (ACT(I,J),J=1,6)
10          CONTINUE
            CALL INIT6(NACT,AXY,ACT,FORCE)

          END IF
          IF(JCTR.EQ.2) GOTO 3000
          IF(JCTR.EQ.3) READ(51,*) INO
3000      READ(51,'(A4)') TAG

c           To read information of [ARRANGE] block
c
          READ(51,'(A9)') TITLE
c   
c         To read the node number of the fundation in the whole
c         bridge structure.
 
          READ(51,*) PNUM,SNUM
          READ(51,*) (PXY(K,1),PXY(K,2),K=1,PNUM)
          IF(SNUM.GT.0) READ(51,*) (SXY(K,1),SXY(K,2),K=1,SNUM)
          READ(51,'(A4)') TAG
c
c           To read information of [NO_SIMU] block
c
          READ(51,'(A9)') TITLE
          READ(51,*) (KCTR(K),K=1,PNUM)
          CALL INIT1(PNUM,KCTR,IDF)
c
c           To read information of <0> segment
c
          READ(51,'(A3)') STAG
          IF(STAG.EQ.'<0>') GOTO 3010
          WRITE(*,'(///9HError:<0>/)')
          STOP
3010        CALL INIT2(0,PNUM)
C           WRITE(*,'(I4)') IDF
c
c           To read information if any KCTR(K)<>0
c
          DO 13 IK=1,IDF-1
            READ(51,'(A1,I2,A1)') CH,IM,CH
c             when KCTR(K) > 0             
            IF(IM.GT.0) CALL INIT2(IM,PNUM)
c             when KCTR(K) < 0
            IF(IM.LT.0) THEN
              READ(51,*) JJ
              DO 12 IA=1,JJ
12              READ(51,*) SIG(IA),JNEW(IA),VNEW(IA)
              CALL INIT4(IM,JJ,PNUM,SIG,JNEW,VNEW)
              ENDIF
13          CONTINUE
          READ(51,'(A4)') TAG
c
c           To read information of [SIMUPILE] block
c
          READ(51,'(A9)') TITLE
          IF(SNUM.EQ.0) GOTO 3020
c           To read if there are any simulative piles
          READ(51,*) (KSCTR(KS),KS=1,SNUM)
          CALL INIT1(SNUM,KSCTR,IDF)
          IS=PNUM*6
          DO 15 IK=1,IDF
            READ(51,'(A1,I2,A1)') CH,IM,CH
            CALL INIT5(IM,IS,SNUM)
15          CONTINUE
3020       READ(51,'(A4)') TAG
c
c           To calculate lengths of piles above ground: ZFR(K)
c                                       & below ground: ZBL(K)
          DO 18 K=1,PNUM
            ZFR(K)=0.0
            ZBL(K)=0.0
            DO 16 IA=1,NFR(K)
16            ZFR(K)=ZFR(K)+HFR(K,IA)
            DO 17 IA=1,NBL(K)
17            ZBL(K)=ZBL(K)+HBL(K,IA)
18          CONTINUE

c      To calculate the whole coordinator of Z of the piles:
c         
c         do 20 i=1,pnum
c         nfb(i)=nfr(i)+nbl(i)+1           
c          write(*,*)' nfb,nfr,nbl,i',nfb(i),nfr(i),nbl(i),i
c         zfb(i,1)=0.0
c         do 21 j=2,nfr(i)
c21       zfb(i,j)=zfb(i,j-1)+hfr(i,j-1)
c         do 22 j=nfr(i)+2,nfb(i)
c22       zfb(i,j)=zfb(i,j-1)+hbl(i,j-nfr(I)-1) 
c20       continue
c
cc       To output the non_simulative piles information to the 
cc       file of [pile.pos].
c
c         write(53,'(i5)') node
c         write(53,'(i5)') pnum
c         write(53,'(6e14.4)')((pxy(i,j),j=1,2),i=1,pnum)
c         do 30 i=1,pnum
c         write(53,'(i5)') nfb(i)
c         write(53,'(6e14.4)')(zfb(i,j),j=1,nfb(i))
c30       continue
c

          RETURN 
          END          

c*****************************************************************
c   Sub to calculate difference number in control information of
c   pile
c*****************************************************************
          SUBROUTINE INIT1(PNUM,KCTR,IDF)
          INTEGER PNUM
          DIMENSION KCTR(PNUM)
          IDF=1
          DO 31 K=2,PNUM
           DO 30 KI=1,K-1
            IF(KCTR(K).EQ.KCTR(KI)) GOTO 31
30         CONTINUE
           IDF=IDF+1
31        CONTINUE
          RETURN
          END               

c*****************************************************************
c       Sub to read the <0> segment information of non_simulative
c       piles
c*****************************************************************          
          SUBROUTINE INIT2(IM,PNUM)
         COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
          DIMENSION AGL1(3),HFR1(15),DOF1(15),HBL1(15),DOB1(15),
     !              PMT1(15),PFI1(15),NSG1(15),NSF1(15)
          INTEGER PNUM
          READ(51,*) KSH1,KSU1,(AGL1(IA),IA=1,3)
          READ(51,*) NFR1,(HFR1(II),DOF1(II),NSF1(II),II=1,NFR1)
          READ(51,*) NBL1,(HBL1(II),DOB1(II),PMT1(II),PFI1(II),
     !               NSG1(II),II=1,NBL1)
          READ(51,*) PMB1,PEH1,PKE1
          DO 43 K=1,PNUM
           CALL INIT3(IM,KCTR(K),KTEST)
           IF(KTEST.EQ.0) GOTO 43
           KSH(K)=KSH1
           KSU(K)=KSU1
           DO 40 IA=1,3
40           AGL(K,IA)=AGL1(IA)
           NFR(K)=NFR1
           DO 41 II=1,NFR(K)
             HFR(K,II)=HFR1(II)
             DOF(K,II)=DOF1(II)
41           NSF(K,II)=NSF1(II)
           NBL(K)=NBL1
           DO 42 II=1,NBL(K)
             HBL(K,II)=HBL1(II)
             DOB(K,II)=DOB1(II)
             PMT(K,II)=PMT1(II)
             PFI(K,II)=PFI1(II)
42           NSG(K,II)=NSG1(II)
           PMB(K)=PMB1
           PEH(K)=PEH1
           PKE(K)=PKE1
43        CONTINUE
          RETURN
          END
   
c***************************************************************
c        Sub to test IM value
c***************************************************************
          SUBROUTINE INIT3(IM,K,KTEST)
          KTEST=0
          IF(IM.EQ.0.AND.K.LE.0) KTEST=1
          IF(IM.GE.0.AND.K.EQ.IM) KTEST=1
          RETURN
          END

c**************************************************************
c    Sub to read <-I> segment information and to modify initial
c    information
c**************************************************************
          SUBROUTINE INIT4(IM,JJ,PNUM,SIG,JNEW,VNEW)
        COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
          CHARACTER*4 SIG(30)
          DIMENSION JNEW(1000),VNEW(1000),NIM(1000)
          INTEGER PNUM
          KK=0
          DO 50 K=1,PNUM
            IF(KCTR(K).NE.IM) GOTO 50
            KK=KK+1
            NIM(KK)=K
50          CONTINUE
          IF(KK.EQ.0) GOTO 69
          DO 70 IA=1,JJ
            IF(SIG(IA).EQ.'KSH=') THEN
              DO 51 IB=1,KK
51              KSH(NIM(IB))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'KSU=') THEN
              DO 52 IB=1,KK
52              KSU(NIM(IB))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'AGL=') THEN
              DO 53 IB=1,KK
53              AGL(NIM(IB),JNEW(IA))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'NFR=') THEN
              DO 54 IB=1,KK
54              NFR(NIM(IB))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'HFR=') THEN
              DO 55 IB=1,KK
55              HFR(NIM(IB),JNEW(IA))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'DOF=') THEN
              DO 56 IB=1,KK
56              DOF(NIM(IB),JNEW(IA))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'NSF=') THEN
              DO 71 IB=1,KK
71              NSF(NIM(IB),JNEW(IA))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'NBL=') THEN
              DO 57 IB=1,KK
57              NBL(NIM(IB))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'HBL=') THEN
              DO 58 IB=1,KK
58              HBL(NIM(IB),JNEW(IA))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'DOB=') THEN
              DO 59 IB=1,KK
59              DOB(NIM(IB),JNEW(IA))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'PMT=') THEN
              DO 60 IB=1,KK
60              PMT(NIM(IB),JNEW(IA))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'PFI=') THEN
              DO 61 IB=1,KK
61              PFI(NIM(IB),JNEW(IA))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'NSG=') THEN
              DO 62 IB=1,KK
62              NSG(NIM(IB),JNEW(IA))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'PMB=') THEN
              DO 63 IB=1,KK
63              PMB(NIM(IB))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'PEH=') THEN
              DO 64 IB=1,KK
64              PEH(NIM(IB))=VNEW(IA)
              GOTO 70
              ENDIF
            IF(SIG(IA).EQ.'PKE=') THEN
              DO 65 IB=1,KK
65              PKE(NIM(IB))=VNEW(IA)
              GOTO 70
            ELSE
69            WRITE(*,'(''Error: <'',I2,1H>)') IM
              STOP
            ENDIF
70        CONTINUE
          RETURN
          END

c************************************************************
c      Sub to read informaion of simulative piles
c************************************************************
          SUBROUTINE INIT5(IM,IS,SNUM)
          COMMON /SIMU/ SXY(20,2),KSCTR(20)
          COMMON /ESTIF/ ESP(1000000,6)
          INTEGER SNUM
          DIMENSION A(6),B(6,6)
          IF(IM.LT.0) THEN
            READ(51,*) (A(IA),IA=1,6)
            DO 62 K=1,SNUM
              IF(KSCTR(K).EQ.IM) THEN
                DO 61 IA=1,6
                  IS=IS+1
                  DO 60 IB=1,6
60                  ESP(IS,IB)=0.0
61                ESP(IS,IA)=A(IA)
              END IF
62            CONTINUE
          END IF
          IF(IM.GT.0) THEN
            READ(51,*) ((B(IA,IB),IB=1,6),IA=1,6)
            DO 64 K=1,SNUM
              IF(KSCTR(K).EQ.IM) THEN
                DO 63 IA=1,6
                  IS=IS+1
                  DO 63 IB=1,6
63                  ESP(IS,IB)=B(IA,IB)
              END IF
64          CONTINUE
          END IF 
          RETURN
          END

c**************************************************************
c        Sub to combine external forces
c**************************************************************
          SUBROUTINE INIT6(NACT,AXY,ACT,FORCE)
          DIMENSION AXY(10,2),ACT(10,6),FORCE(6),TU(6,6),TN(6,6),
     !         A(6),B(6)
          DO 65 IF=1,6
65          FORCE(IF)=0.0
          DO 68 IF=1,NACT
            DO 66 I=1,6
66            A(I)=ACT(IF,I)
            CALL TMATX(AXY(IF,1),AXY(IF,2),TU)
            CALL TRNSPS(6,6,TU,TN)
            CALL MULULT(6,6,1,TN,A,B)
            DO 67 I=1,6
67            FORCE(I)=FORCE(I)+B(I)
68            CONTINUE
          RETURN
          END

c
c***************************************************************
c   Sub to calculate deformation factors of piles
c***************************************************************
c
          SUBROUTINE BTXY(PNUM,ZFR,ZBL,BTX,BTY)
        COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
          INTEGER PNUM
          REAL KINF(2),KA
          DIMENSION ZFR(PNUM),ZBL(PNUM),BTX(1000,15),
     !	          BTY(1000,15),GXY(1000,2)
          DO 110 K=1,PNUM
            GXY(K,1)=PXY(K,1)+ZFR(K)*AGL(K,1)
110         GXY(K,2)=PXY(K,2)+ZFR(K)*AGL(K,2)
          DO 111 K=1,PNUM
            DO 111 K1=K+1,PNUM
              S=SQRT((GXY(K,1)-GXY(K1,1))**2+(GXY(K,2)-
     !          GXY(K1,2))**2)-(DOB(K,1)+DOB(K1,1))/2.0
              IF(S.LT.1.0) THEN
                CALL KINF1(1,PNUM,DOB,ZBL,GXY,KINF(1))
                CALL KINF1(2,PNUM,DOB,ZBL,GXY,KINF(2))
                GOTO 2100
              END IF
111       CONTINUE
          CALL KINF2(1,PNUM,DOB,ZBL,GXY,KINF(1))
          CALL KINF2(2,PNUM,DOB,ZBL,GXY,KINF(2))
C2100      WRITE(*,'(6HKINF= ,2F10.4)') KINF(1),KINF(2)
2100          DO 115 K=1,PNUM
            IF(K.GT.1) THEN
              DO 113 K1=1,K-1
                IF(KCTR(K).EQ.KCTR(K1)) THEN
                  DO 112 IA=1,NBL(K1)
                    BTX(K,IA)=BTX(K1,IA)
112                 BTY(K,IA)=BTY(K1,IA)
                  GOTO 115
                END IF
113           CONTINUE
            END IF
            KA=1.0
            IF(KSH(K).EQ.1) KA=0.9
            DO 114 IA=1,NBL(K)
              BX1=KA*KINF(1)*(DOB(K,IA)+1.0)
              BY1=KA*KINF(2)*(DOB(K,IA)+1.0)
              CALL EAJ(KSH(K),PKE(K),DOB(K,IA),A,B)
              BTX(K,IA)=(PMT(K,IA)*BX1/(PEH(K)*B))**0.2
114           BTY(K,IA)=(PMT(K,IA)*BY1/(PEH(K)*B))**0.2
115       CONTINUE
C          WRITE(*,'(5I2)') (NBL(K),K=1,PNUM)
C          WRITE(*,'(4HBTX=,5E12.4)') ((BTX(K,IA),IA=1,NBL(K)),K=1,PNUM)
C          WRITE(*,'(4HBTY=,5E12.4)') ((BTY(K,IA),IA=1,NBL(K)),K=1,PNUM)
          RETURN
          END
          
c*******************************************************************
c        Sub to calculate influential factor 
c******************************************************************* 
c   This sub treats with situation where spacing between piles is 
c   less than 1.0 m. 
          SUBROUTINE KINF1(IM,PNUM,DOB,ZBL,GXY,KINF)
          INTEGER PNUM
          REAL KINF
          DIMENSION DOB(PNUM),ZBL(PNUM),GXY(1000,2),
     !              AA(1000),DD(1000),ZZ(1000)
          IN=1
          AA(1)=GXY(1,IM)
          DD(1)=DOB(1)
          ZZ(1)=ZBL(1)
          DO 121 K=2,PNUM
            DO 120 K1=1,K-1
              IF(GXY(K,IM).EQ.GXY(K1,IM)) GOTO 121
120         CONTINUE
            IN=IN+1
            AA(IN)=GXY(K,IM)
            DD(IN)=DOB(K)
            ZZ(IN)=ZBL(K)
121       CONTINUE
          CALL KINF3(IN,AA,DD,ZZ,KINF)
          RETURN
          END

c*******************************************************************
c        Sub to calculate influential factor 
c******************************************************************* 
c   This sub treats with situation where spacing between piles is 
c   greater than 1.0 m. 
          SUBROUTINE KINF2(IM,PNUM,DOB,ZBL,GXY,KMIN)
          INTEGER PNUM
          REAL KINF,KMIN
          DIMENSION DOB(PNUM),ZBL(PNUM),GXY(1000,2),NIN(1000),
     !              IN(1000),NOK(1000,1000),AA(1000),DD(1000),ZZ(1000)
          IM1=1
          IF(IM.EQ.1) IM1=2
          NROW=0
          DO 131 K=1,PNUM
            DO 130 K1=1,K-1
              IF(GXY(K,IM1).EQ.GXY(K1,IM1)) THEN
                NA=NIN(K1)
                IN(NA)=IN(NA)+1
                NOK(NA,IN(NA))=K
                GOTO 131
              END IF    
130         CONTINUE
            NROW=NROW+1
            NIN(K)=NROW
            IN(NROW)=1
            NOK(NROW,1)=K
131       CONTINUE
          KMIN=1.0
          DO 133 I=1,NROW
            DO 132 J=1,IN(I)
              K=NOK(I,J)
              AA(J)=GXY(K,IM)
              DD(J)=DOB(K)
132           ZZ(J)=ZBL(K)
            CALL KINF3(IN(I),AA,DD,ZZ,KINF)
            IF(KINF.LT.KMIN) KMIN=KINF
133       CONTINUE
          RETURN
          END

c******************************************************
c   Sub to calculate influential factor of a pile row
c******************************************************
          SUBROUTINE KINF3(IN,AA,DD,ZZ,KINF)
          REAL KINF
          DIMENSION AA(IN),DD(IN),ZZ(IN),HO(1000)
          IF(IN.EQ.1) THEN
            KINF=1.0
            GOTO 2200
          END IF
          DO 140 I=1,IN
            HO(I)=3.0*(DD(I)+1.0)
            IF(HO(I).GT.ZZ(I)) HO(I)=ZZ(I)
140       CONTINUE
          LO=100.0
          DO 141 I=1,IN
            DO 141 I1=I+1,IN
              S=ABS(AA(I)-AA(I1))-(DD(I)+DD(I1))/2.0
              IF(S.LT.LO) THEN
                LO=S
                HOO=HO(I)
                IF(HOO.LT.HO(I1)) HOO=HO(I1)
              END IF
141       CONTINUE  
          IF(LO.GE.0.6*HOO) THEN
            KINF=1.0
          ELSE
            CALL PARC(IN,C)
            KINF=C+(1.0-C)*LO/(0.6*HOO)
          END IF
2200      RETURN
          END
          
c*****************************************************
c     Sub to give the pile group coefficient of Kinf
c*****************************************************
          SUBROUTINE PARC(IN,C)
          IF(IN.EQ.1) C=1           
          IF(IN.EQ.2) C=0.6
          IF(IN.EQ.3) C=0.5
          IF(IN.GE.4) C=0.45       
          RETURN
          END




c****************************************************************
c               Sub to form the program head
c****************************************************************
c
      SUBROUTINE HEAD1
      WRITE(*,'(//////)')
      WRITE(*,990)
      WRITE(*,1000)
      WRITE(*,1010)
      WRITE(*,1020)   
      WRITE(*,1030)
      WRITE(*,1040)
      WRITE(*,1050)
      WRITE(*,1060)
      WRITE(*,1070)
      WRITE(*,1080)
990   FORMAT(/5X,'Welcome to use the BCAD_PILE program !!'/)
1000  FORMAT(5X,'  This program is aimed to execute spatial statical ana
     !lysis of pile')
1010  FORMAT(5X,'foundations of bridge substructures. If you have any qu
     !estions about')
1020  FORMAT(5X,'this program, please do not hesitate to write to :'/)
1030  FORMAT(50X,'CAD Reseach Group')
1040  FORMAT(50X,'Dept.of Bridge Engr.')
1050  FORMAT(50X,'Tongji University')
1060  FORMAT(50X,'1239 Sipin Road ')
1070  FORMAT(50X,'Shanghai 200092')
1080  FORMAT(50X,'P.R.of China'/)
      RETURN
      END


c****************************************************************
c               Sub to form the program head
c****************************************************************
c
      SUBROUTINE HEAD2
      WRITE(52,'(//////)')
      WRITE(52,980)
      WRITE(52,965)
      WRITE(52,900)
      WRITE(52,910) 
      WRITE(52,920)
      WRITE(52,930)
      WRITE(52,940)
      WRITE(52,950)
      WRITE(52,960)
      WRITE(52,965) 
      WRITE(52,970)
      WRITE(52,965)
      WRITE(52,980)
      WRITE(52,990)
      WRITE(52,1000)
      WRITE(52,1010)
      WRITE(52,1020)   
      WRITE(52,1030)
      WRITE(52,1040)
      WRITE(52,1050)
      WRITE(52,1060)
      WRITE(52,1070)
      WRITE(52,1080)
900   FORMAT(7X,'+    BBBBBB       CCCC        A       DDDDD         PPPP
     -PP     III     L         EEEEEEE    +')
910   FORMAT(7X,'+    B     B     C    C      A A      D    D        P  
     -  P     I      L         E          +')
920   FORMAT(7X,'+    B     B    C           A   A     D     D       P  
     -  P     I      L         E          +')
930   FORMAT(7X,'+    BBBBBB     C          A     A    D     D       PPPP
     -PP      I      L         EEEEEEE    +')
940   FORMAT(7X,'+    B     B    C          AAAAAAA    D     D       P  
     -        I      L         E          +')
950   FORMAT(7X,'+    B     B     C    C    A     A    D    D        P   
     -        I      L     L   E          +')
960   FORMAT(7X,'+    BBBBBB       CCCC     A     A    DDDDD   ===== P   
     -       III      LLLLL    EEEEEEE    +')
965   FORMAT(7X,'+                                                       
     -                                    +')
970   FORMAT(7X,'+                        Copyright 1990, Version 
     -1.10  modfied by Zhiq Wang                  +') 
980   FORMAT(7X,'++++++++++++++++++++++++++++++++++++++++++++++++++++++++
     -+++++++++++++++++++++++++++++++++++++')
990   FORMAT(/15X,'Welcome to use the BCAD_PILE program !!'/)
1000  FORMAT(15X,'This program is aimed to execute spatial statical ana
     !lysis of pile')
1010  FORMAT(15X,'foundations of bridge substructures. If you have any q
     !uestions about')
1020  FORMAT(15X,'this program, please do not hesitate to write to :'/)
1030  FORMAT(60X,'CAD Reseach Group')
1040  FORMAT(60X,'Dept.of Bridge Engr.')
1050  FORMAT(60X,'Tongji University')
1060  FORMAT(60X,'1239 Sipin Road ')
1070  FORMAT(60X,'Shanghai 200092')
1080  FORMAT(60X,'P.R.of China'/)
1090  FORMAT(7X,'*******************************************************
     -************************************'/)
      RETURN
      END


c
c**************************************************************
c    Sub to calculate areas at the bottom of piles
c**************************************************************
c
          SUBROUTINE AREA(PNUM,ZFR,ZBL,AO)
        COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
          INTEGER PNUM
          DIMENSION ZFR(PNUM),ZBL(PNUM),AO(PNUM),BXY(1000,2),
     !              W(1000),SMIN(1000)
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
90          X=X+HFR(K,IA)/(PEH(K)*A)
          DO 91 IA=1,NBL(K)
            CALL EAJ(KSH(K),PKE(K),DOB(K,IA),A,B)
91          X=X+PKC*HBL(K,IA)/(PEH(K)*A)
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

c***************************************************************
c     Sub to calculate axial stiffness of each pile
c***************************************************************
c
          SUBROUTINE STIFF_N(PNUM,ZFR,ZBL,AO,RZZ)
        COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
          INTEGER PNUM
          DIMENSION RZZ(PNUM),AO(PNUM),ZFR(PNUM),ZBL(PNUM)
         CALL STN(1,ZBL(1),AO(1),RZZ(1))
          DO 101 K=2,PNUM
            DO 100 IA=1,K-1
              IF(KCTR(K).EQ.KCTR(IA).AND.AO(K).EQ.AO(IA)) THEN
                RZZ(K)=RZZ(IA)
                GOTO 101
              END IF
100         CONTINUE
            CALL STN(K,ZBL(K),AO(K),RZZ(K))
101       CONTINUE
          RETURN
          END
          

c*****************************************************************
c   Sub to calculte relational matrix of free segments ofpiles
c*****************************************************************
c
          SUBROUTINE RLTFR(NFR,EJ,HFR,KFR)
          DIMENSION EJ(NFR),HFR(NFR),KFR(4,4),R(4,4),RM(4,4)
          REAL KFR
          CALL MFREE(EJ(1),HFR(1),KFR)
          DO 172 IA=2,NFR
            CALL MFREE(EJ(IA),HFR(IA),R)
            CALL MULULT(4,4,4,KFR,R,RM)
            DO 171 I=1,4
              DO 171 J=1,4
171             KFR(I,J)=RM(I,J)
172       CONTINUE
C          WRITE(*,'(5HKFR= ,4E12.4)') ((KFR(I,J),J=1,4).I=1,4)
          RETURN
          END
          
c*************************************************************
c  Sub to calculate relational matrix of one pile segment
c*************************************************************
c
          SUBROUTINE MFREE(EJ,H,R)
          DIMENSION R(4,4)
          DO 181 I=1,4
            DO 180 J=1,4
180           R(I,J)=0.0
181         R(I,I)=1.0
          R(1,2)=H
          R(1,3)=H**3/(6.0*EJ)
          R(1,4)=-H**2/(2.0*EJ)
          R(2,3)=H**2/(2.0*EJ)
          R(2,4)=-H/EJ
          R(4,3)=-H
          RETURN
          END
          
c************************************************************
c   Sub to combine relational matrics of free and non-free 
c   pile segments
c************************************************************ 
c
          SUBROUTINE COMBX(KBX,KFR,KX)
          REAL KBX,KFR,KX,KV,KF
          DIMENSION KBX(4,4),KFR(4,4),KX(4,4),KV(4,4),KF(4,4)
C          WRITE(*,'(4E12.4)') ((KFR(I,J),J=1,4),I=1,4)
          DO 190 I=1,4
190         KBX(I,4)=-KBX(I,4)
          CALL MULULT(4,4,4,KBX,KFR,KX)
          RETURN
          END
          
c***********************************************************
c   Sub to calculate element lateral stiffnesses of a pile
c***********************************************************
c
          SUBROUTINE CNDTN(KSU,KX,KY,RZZ,KE)
          DIMENSION AT(2,2)
          REAL KX(4,4),KY(4,4),KE(6,6)
          DO 200 I=1,6
            DO 200 J=1,6
200            KE(I,J)=0.0
          CALL DVSN(KSU,KX,AT)
C           WRITE(*,'(4HATX=,2E12.4)') ((AT(I,J),J=1,2),I=1,2)
          KE(1,1)=AT(1,1)
          KE(1,5)=AT(1,2)
          KE(5,1)=AT(2,1)
          KE(5,5)=AT(2,2)
          CALL DVSN(KSU,KY,AT)
          KE(2,2)=AT(1,1)
          KE(2,4)=-AT(1,2)
          KE(4,2)=-AT(2,1)
          KE(4,4)=AT(2,2)
          KE(3,3)=RZZ
          KE(6,6)=0.1*(KE(4,4)+KE(5,5))
C          WRITE(*,'(6E12.4)') ((KE(I,J),J=1,6),I=1,6)
          RETURN
          END

c**********************************************************
c    Sub to treat with boundary conditions of piles
c**********************************************************
c
          SUBROUTINE DVSN(KSU,KXY,AT)
          DIMENSION AT(2,2),A11(2,2),A12(2,2),A21(2,2),
     !        A22(2,2),AV(2,2)
          REAL KXY(4,4)
          DO 210 I=1,2
            DO 210 J=1,2
              A11(I,J)=KXY(I,J)
              A12(I,J)=KXY(I,J+2)
              A21(I,J)=KXY(I+2,J)
210           A22(I,J)=KXY(I+2,J+2)
          IF(KSU.EQ.4) THEN
            CALL SINVER(A12,2,AV,JE)
            CALL MULULT(2,2,2,AV,A11,AT)
          ELSE
            CALL SINVER(A22,2,AV,JE)
            CALL MULULT(2,2,2,AV,A21,AT)
          END IF
          DO 220 I=1,2
            DO 220 J=1,2
220           AT(I,J)=-AT(I,J)
          RETURN
          END

c******************************************************
c     Sub to form transform matrix of piles
c******************************************************
c
          SUBROUTINE TRNSFR(X,Y,Z,TK)
          DIMENSION TK(6,6)
          B=SQRT(Y**2+Z**2)
          TK(1,1)=B
          TK(1,2)=0.0
          TK(1,3)=X
          TK(2,1)=-X*Y/B
          TK(2,2)=Z/B
          TK(2,3)=Y
          TK(3,1)=-X*Z/B
          TK(3,2)=-Y/B
          TK(3,3)=Z
          DO 220 IA=1,3
            DO 220 IB=1,3
              TK(IA,IB+3)=0.0
              TK(IA+3,IB+3)=TK(IA,IB)
220           TK(IA+3,IB)=0.0
          RETURN
          END
          
c******************************************************
c         Sub to transfer a matrix
c******************************************************
c
          SUBROUTINE TRNSPS(M,N,B,BT)
          DIMENSION B(M,N),BT(N,M)
          DO 230 I=1,M
            DO 230 J=1,N
230           BT(J,I)=B(I,J)
          RETURN
          END
         

c*****************************************************************
c        Sub to calculate element stiffnesses of piles 
c*****************************************************************
c
          SUBROUTINE PSTIFF(PNUM,RZZ,BTX,BTY)
        COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
          COMMON /ESTIF/ ESP(1000000,6)
          INTEGER PNUM
          DIMENSION BTX(1000,15),BTY(1000,15),BT1(15),BT2(15),EJ(15),
     !         H(1006),RZZ(PNUM)
          REAL KBX(4,4),KBY(4,4),KFR(4,4),KX(4,4),KY(4,4),KE(6,6)
          DO 155 K=1,PNUM
            IF(NBL(K).EQ.0) THEN
              DO 149 I=1,4
                DO 148 J=1,4
                  KBX(I,J)=0.0
148               KBY(I,J)=0.0
                KBX(I,I)=1.0
149             KBY(I,I)=1.0                
              GOTO  2055
            END IF   
            H(1)=0.0
            DO 150 IA=1,NBL(K)
              BT1(IA)=BTX(K,IA)
              BT2(IA)=BTY(K,IA)
              CALL EAJ(KSH(K),PKE(K),DOB(K,IA),A,B)
              EJ(IA)=PEH(K)*B
150           H(IA+1)=H(IA)+HBL(K,IA)
            CALL RLTMTX(NBL(K),BT1,BT2,EJ,H,KBX,KBY)
C             WRITE(*,'(5HKBX= ,4E12.4)') ((KBX(I,J),J=1,4),I=1,4)
C             WRITE(*,'(5HKBY= ,4E12.4)') ((KBY(I,J),J=1,4),I=1,4)  
2055          IF(NFR(K).EQ.0) THEN
              DO 152 I=1,4
                DO 151 J=1,4
                  KX(I,J)=KBX(I,J)
151               KY(I,J)=KBY(I,J)
                KX(I,4)=-KX(I,4)
152             KY(I,4)=-KY(I,4)
              GOTO 2060
            END IF
            DO 153 IA=1,NFR(K)
              CALL EAJ(KSH(K),PKE(K),DOF(K,IA),A,B)
              EJ(IA)=PEH(K)*B
153           H(IA)=HFR(K,IA)
            CALL RLTFR(NFR(K),EJ,H,KFR)
            CALL COMBX(KBX,KFR,KX)
            CALL COMBX(KBY,KFR,KY)
C            WRITE(*,'(/32H**Element stiffness at the top**)')
C            WRITE(*,'(4E12.4,5X,4E12.4)') ((KX(I,J),J=1,4),
C     !       (KY(I,J),J=1,4),I=1,4)
2060        CALL CNDTN(KSU(K),KX,KY,RZZ(K),KE)
C            WRITE(*,'(24H/** The stiffness of No. ,I2,8H pile **/)') K
            DO 154 I=1,6
C              WRITE(*,'(6E12.4)') (KE(I,J),J=1,6)
              DO 154 J=1,6
                K1=(K-1)*6
154             ESP(K1+I,J)=KE(I,J)
155       CONTINUE
          RETURN
          END


c****************************************************************
c     Sub to calculate relational matrics of non-free pile
c     segments
c****************************************************************
c
          SUBROUTINE RLTMTX(NBL,BT1,BT2,EJ,H,KBX,KBY)
          DIMENSION BT1(NBL),BT2(NBL),EJ(NBL),H(NBL+1),KBX(4,4),
     !       KBY(4,4),A1(4,4),A2(4,4),A3(4,4)
          REAL KBX,KBY
C           WRITE(*,'(2E12.4)') (BT1(I1),BT2(I1),I1=1,NBL)
          CALL SAA(BT1(1),EJ(1),H(1),H(2),KBX)
          DO 161 IA=2,NBL
            DO 160 I1=1,4
              DO 160 J1=1,4
160             A1(I1,J1)=KBX(I1,J1)
            CALL SAA(BT1(IA),EJ(IA),H(IA),H(IA+1),A2)
            CALL MULULT(4,4,4,A2,A1,KBX)
161       CONTINUE  
          DO 162 IA=1,NBL
            IF(ABS(BT2(IA)-BT1(IA)).GT.1.0E-10) GOTO 2300
162       CONTINUE
          DO 163 I1=1,4
            DO 163 J1=1,4
163           KBY(I1,J1)=KBX(I1,J1)
          GOTO 2400
2300      CALL SAA(BT2(1),EJ(1),H(1),H(2),KBY)
          DO 165 IA=2,NBL
            DO 164 I1=1,4
              DO 164 J1=1,4
164             A1(I1,J1)=KBY(I1,J1)
            CALL SAA(BT2(IA),EJ(IA),H(IA),H(IA+1),A2)
            CALL MULULT(4,4,4,A2,A1,KBY)
165       CONTINUE
2400      RETURN
          END

c************************************************************
c  Sub to calculate relational matrix of one non-free pile
c  segment
c************************************************************
c
          SUBROUTINE SAA(BT,EJ,H1,H2,AI)
          DIMENSION AI(4,4),AI1(4,4),AI2(4,4),AI3(4,4)
          CALL PARAM(BT,EJ,H1,AI1)
          CALL PARAM(BT,EJ,H2,AI2)
          CALL SINVER(AI1,4,AI3,JE)
C          WRITE(*,'(2HH=,2F10.4)') H1,H2
C          WRITE(*,'(8E12.4)') ((AI1(I,J),J=1,4),(AI2(I,J),
C     !         J=1,4),I=1,4)
          CALL MULULT(4,4,4,AI2,AI3,AI)
          DO 167 I=1,2
            DO 167 J=1,2
              AI(I,J+2)=AI(I,J+2)/EJ
167           AI(I+2,J)=AI(I+2,J)*EJ
          DO 168 J=1,4
            X=AI(3,J)
            AI(3,J)=AI(4,J)
168         AI(4,J)=X
          DO 169 I=1,4
            X=AI(I,3)
            AI(I,3)=AI(I,4)
169         AI(I,4)=X
C          WRITE(*,'(//4E12.4/)') BT,EJ,H1,H2
C          WRITE(*,'(4E12.4)') ((AI(I,J),J=1,4),I=1,4)
          RETURN
          END
           
c*********************************************************
c  Sub to give the value of a coefficient matrix
c*********************************************************
c
          SUBROUTINE PARAM(BT,EJ,X,AA)
          DIMENSION AA(4,4)
          Y=BT*X
          IF(Y.GT.6.0) Y=6.0
          CALL PARAM1(Y,A1,B1,C1,D1,A2,B2,C2,D2)
          CALL PARAM2(Y,A3,B3,C3,D3,A4,B4,C4,D4)
          AA(1,1)=A1
          AA(1,2)=B1/BT
          AA(1,3)=2*C1/BT**2   
          AA(1,4)=6*D1/BT**3
          AA(2,1)=A2*BT
          AA(2,2)=B2
          AA(2,3)=2*C2/BT
          AA(2,4)=6*D2/BT**2
          AA(3,1)=A3*BT**2
          AA(3,2)=B3*BT
          AA(3,3)=2*C3
          AA(3,4)=6*D3/BT
          AA(4,1)=A4*BT**3
          AA(4,2)=B4*BT**2
          AA(4,3)=2*C4*BT
          AA(4,4)=6*D4
          RETURN
          END
c
c********************************************************************
c    Sub to approximately calculate the value of power series items
c********************************************************************
c
        SUBROUTINE PARAM1(Y,A1,B1,C1,D1,A2,B2,C2,D2)
        A1=1-Y**5/120.0+Y**10/6.048E5-Y**15/1.9813E10+Y**20/2.3038E15
     1     -Y**25/6.9945E20
        B1=Y-Y**6/360.0+Y**11/2851200-Y**16/1.245E11+Y**21/1.7889E16
     1     -Y**26/6.4185E21
         C1=Y**2/2.0-Y**7/1680+Y**12/1.9958E7-Y**17/1.14E12+Y**22/2.0E17
     1     -Y**27/8.43E22
         D1=Y**3/6.0-Y**8/10080+Y**13/1.7297E8-Y**18/1.2703E13
     1     +Y**23/2.6997E18-Y**28/1.33E24
         A2=-Y**4/24.0+Y**9/6.048E4-Y**14/1.3209E9+Y**19/1.1519E14
     1     -Y**24/2.7978E19
         B2=1-Y**5/60.0+Y**10/2.592E5-Y**15/7.7837E9+Y**20/8.5185E14
     1     -Y**25/2.4686E20
         C2=Y-Y**6/240.0+Y**11/1.6632E6-Y**16/6.7059E10+Y**21/9.0973E15
     1     -Y**26/3.1222E21
         D2=Y**2/2-Y**7/1260+Y**12/1.3305E7-Y**17/7.0572E11
     1     +Y**22/1.1738E17-Y**27/4.738E22
         RETURN
         END

c********************************************************************
c    Sub to approximately calculate the value of power series items
c********************************************************************
c
          SUBROUTINE PARAM2(Y,A3,B3,C3,D3,A4,B4,C4,D4)
          A3=-Y**3/6+Y**8/6.72E3-Y**13/9.435E7+Y**18/6.0626E12
     1     -Y**23/1.1657E18
          B3=-Y**4/12+Y**9/25920-Y**14/5.1892E8+Y**19/4.2593E13
     1     -Y**24/9.8746E18
          C3=1-Y**5/40+Y**10/151200-Y**15/4.1912E9+Y**20/4.332E14
     1     -Y**25/1.2009E20
          D3=Y-Y**6/180+Y**11/1108800-Y**16/4.1513E10+Y**21/5.3354E15
     1     -Y**26/1.7543E21
          A4=-Y**2/2+Y**7/840-Y**12/7.257E6+Y**17/3.3681E11
     1     -Y**22/5.0683E16
          B4=-Y**3/3+Y**8/2880-Y**13/3.7066E7+Y**18/2.2477E12
     1     -Y**23/4.1144E17
          C4=-Y**4/8+Y**9/1.512E4-Y**14/2.7941E8+Y**19/2.166E13
     1     -Y**24/4.8034E18
          D4=1-Y**5/30+Y**10/100800-Y**15/2.5946E9+Y**20/2.5406E14
     1     -Y**25/6.7491E19
           RETURN
           END

c*****************************************************************
c Sub to calculate the multification of two matrics.
c*****************************************************************
c
        SUBROUTINE MULULT (M,L,N,A,B,C)
        DIMENSION A(M,L),B(L,N),C(M,N)
        DO 10 I=1,M
        DO 10 K=1,N
          C(I,K)=0.0
          DO 10 J=1,L
10          C(I,K)=C(I,K)+A(I,J)*B(J,K)
        RETURN
        END
c
c*****************************************************************
c Sub to calculate the inverse of a square matrix
c*****************************************************************
c
      SUBROUTINE SINVER(A,N,AMINUS,NER)
      DIMENSION A(N,N),AMINUS(N,N)
      SUM=0.0
      DO 10 I=1,N
        SUM=SUM+ABS(A(I,I))
        DO 20 J=1,N
   20   AMINUS(I,J)=0.0E0
   10 AMINUS(I,I)=1.0
      SUM=SUM/N
C      WRITE(*,*) 'SUM=',SUM
C SUM---average value of the diagonal elements.
C Reduction
      DO 30 IE=1,N-1
        DO 40 I=IE+1,N
          IF(ABS(A(IE,IE)/SUM).LT.1.0E-12) THEN 
            NER=100
            RETURN
          ENDIF
          FCTR=A(I,IE)/A(IE,IE)
          DO 50 J=IE,N
   50     A(I,J)=A(I,J)-A(IE,J)*FCTR
          DO 60 J=1,N
   60     AMINUS(I,J)=AMINUS(I,J)-AMINUS(IE,J)*FCTR
   40   CONTINUE
   30 CONTINUE
C Back substitution
      DO 70 I=1,N
   70 AMINUS(N,I)=AMINUS(N,I)/A(N,N)
      DO 80 IE=N,2,-1
        DO 80 J=1,N
          DO 90 I=1,IE-1
   90     AMINUS(I,J)=AMINUS(I,J)-A(I,IE)*AMINUS(IE,J)
          AMINUS(IE-1,J)=AMINUS(IE-1,J)/A(IE-1,IE-1)
   80 CONTINUE
C      DO 100 I=1,N
C  100 WRITE(*,*) (AMINUS(I,J),J=1,N)
      RETURN
      END 

          
c*********************************************************************
c  Sub to calculate displacements of the cap of pile foundation
c*********************************************************************
c
          SUBROUTINE DISP(JCTR,INO,PNUM,SNUM,PXY,SXY,AGL,FORCE,DUK,SO)
          INTEGER PNUM,SNUM
          COMMON /ESTIF/ ESP(1000000,6)
          DIMENSION PXY(1000,2),SXY(20,2),AGL(1000,3),FORCE(6),
     !              TU(6,6),TN(6,6),B(6,6),SO(6,6),DUK(1000,6),
     !              TK(6,6),TK1(6,6),A(6,6),A1(6,6),C(6),C1(6)

          IF(JCTR.EQ.3) THEN
c     Only to calculate stiffness of one pointed pile 
             WRITE(52,710) INO
             WRITE(52,'(6E12.4)') ((ESP((INO-1)*6+I,J),J=1,6),I=1,6)
             STOP

           DO 100 IA=1,6
            DO 100 IB=1,6 
             SO(IA,IB)=ESP((INO-1)*6+IA,IB)  
100        CONTINUE
           RETURN

          END IF
          DO 200 IA=1,6
            DO 200 IB=1,6
200           SO(IA,IB)=0.0
          DO 203 K=1,PNUM+SNUM
              DO 201 IA=1,6
                DO 201 IB=1,6
201               A(IA,IB)=ESP((K-1)*6+IA,IB)
            IF(K.LE.PNUM) THEN
              CALL TRNSFR(AGL(K,1),AGL(K,2),AGL(K,3),TK)
              CALL TRNSPS(6,6,TK,TK1)
              CALL MULULT(6,6,6,TK,A,A1)
              CALL MULULT(6,6,6,A1,TK1,A)
              X=PXY(K,1)
              Y=PXY(K,2)
            ELSE
              K1=K-PNUM
              X=SXY(K1,1)
              Y=SXY(K1,2)
            END IF
            CALL TMATX(X,Y,TU)
            CALL TRNSPS(6,6,TU,TN)
            CALL MULULT(6,6,6,A,TU,B)
            CALL MULULT(6,6,6,TN,B,A)
            DO 202 IA=1,6
              DO 202 IB=1,6
202             SO(IA,IB)=SO(IA,IB)+A(IA,IB)
203         CONTINUE

c           WRITE(*,'(6E12.4)') ((SO(I,J),J=1,6),I=1,6)
          IF(JCTR.EQ.2) THEN
c       Only to calculate stiffness of the entire pile foundation 
            WRITE(52,700)
            WRITE(52,'(6E12.4)') ((SO(I,J),J=1,6),I=1,6)
            STOP   
           RETURN

          END IF
          CALL GAOS(6,SO,FORCE)
          WRITE(52,800)
          WRITE(52,801)
          WRITE(52,800)
          WRITE(52,803) FORCE(1)
          WRITE(52,804) FORCE(2)
          WRITE(52,805) FORCE(3)
          WRITE(52,806) FORCE(4)
          WRITE(52,807) FORCE(5)
          WRITE(52,808) FORCE(6)
          DO 206 K=1,PNUM
            CALL TMATX(PXY(K,1),PXY(K,2),TU)
            CALL MULULT(6,6,1,TU,FORCE,C1)
              CALL TRNSFR(AGL(K,1),AGL(K,2),AGL(K,3),TK)
              CALL TRNSPS(6,6,TK,TK1)
              CALL MULULT(6,6,1,TK1,C1,C)
            DO 205 IA=1,6
205           DUK(K,IA)=C(IA)
206       CONTINUE
700    FORMAT(//7X,'*** Stiffness of the entire pile foundation ***'/)
710    FORMAT(//7X,'*** Stiffness of the No.',I2,' pile ***'/)
800     FORMAT(7X,'*****************************************************
     !**************************')
801     FORMAT(15X,'DISPLACEMENTS AT THE CAP CENTER OF PILE FOUNDATION')
803       FORMAT(/16X,'Movement in the direction of X axis : UX=',
     !  E12.4,' (m)')
804       FORMAT(16X,'Movement in the direction of Y axis : UY=',
     !  E12.4,' (m)')
805       FORMAT(16X,'Movement in the direction of Z axis : UZ=',
     !  E12.4,' (m)')
806       FORMAT(16X,'Rotational angle  around X axis :     SX=',
     !  E12.4,' (rad)')
807       FORMAT(16X,'Rotational angle around Y axis :      SY=',
     !  E12.4,' (rad)')
808       FORMAT(16X,'Rotational angle around Z axis :      SZ=',
     !  E12.4,' (rad)'//)
          RETURN
          END
 
c**************************************************************
c  Sub to calculate transform matrix of elemental coordination
c  system
c**************************************************************
c
          SUBROUTINE TMATX(X,Y,TU)
          DIMENSION TU(6,6)
            DO 211 IA=1,6
              DO 210 IB=1,6
210             TU(IA,IB)=0.0
211           TU(IA,IA)=1.0
            TU(1,6)=-Y
            TU(2,6)=X
            TU(3,4)=Y
            TU(3,5)=-X
          RETURN
          END


c********************************************************************
c  Sub to calculte displacements and internal forces of pile bodies 
c********************************************************************
          SUBROUTINE EFORCE(PNUM,BTX,BTY,ZBL,DUK)
        COMMON /PINF/ PXY(1000,2),KCTR(1000),KSH(1000),KSU(1000),
     !    AGL(1000,3),NFR(1000),HFR(1000,15),DOF(1000,15),NSF(1000,15),
     !    NBL(1000),HBL(1000,15),DOB(1000,15),PMT(1000,15),PFI(1000,15),
     !    NSG(1000,15),PMB(1000),PEH(1000),PKE(1000)
C
          COMMON /ESTIF/ ESP(1000000,6)
C
         DIMENSION BTX(1000,15),BTY(1000,15),DUK(1000,6),CE(6),SE(6,6),
     !       PE(6),ZH(100),FX(100,4),FY(100,4),FZ(100),R(4,4),
     !       XA(4),XB(4),XC(4),XD(4),PSX(100),PSY(100)
          INTEGER PNUM

c        To output the data of the pile plan position.

           write(53,'(i5)') pnum
           write(53,'(6e14.4)') ((pxy(i,j),j=1,2),i=1,pnum)

          DO 307 K=1,PNUM
            DO 300 I=1,6
              CE(I)=DUK(K,I)
              DO 300 J=1,6
300             SE(I,J)=ESP((K-1)*6+I,J)
            CALL MULULT(6,6,1,SE,CE,PE)
C            WRITE(*,'(6E12.4/)') (CE(I),I=1,6)
C            WRITE(*,'(6E12.4)') ((SE(I,J),J=1,6),I=1,6)
            ZH(1)=0.0
            FX(1,1)=CE(1)
            FX(1,2)=CE(5)
            FX(1,3)=PE(1)
            FX(1,4)=PE(5)
            FY(1,1)=CE(2)
            FY(1,2)=CE(4)
            FY(1,3)=PE(2)
            FY(1,4)=PE(4)
            FZ(1)=PE(3)
            NSUM=1
            DO 303 IA=1,NFR(K)
              HL=HFR(K,IA)/NSF(K,IA)
              CALL EAJ(KSH(K),PKE(K),DOF(K,IA),A,B)
              EJ=PEH(K)*B
              CALL MFREE(EJ,HL,R)
              DO 303 IN=1,NSF(K,IA)
                DO 301 I=1,4
                  XA(I)=FX(NSUM,I)
301               XC(I)=FY(NSUM,I)
                XC(2)=-XC(2)
                XC(4)=-XC(4)
                CALL MULULT(4,4,1,R,XA,XB)
                CALL MULULT(4,4,1,R,XC,XD)
                NSUM=NSUM+1
                DO 302 I=1,4
                  FX(NSUM,I)=XB(I)
302               FY(NSUM,I)=XD(I)
                FY(NSUM,2)=-XD(2)
                FY(NSUM,4)=-XD(4)
                ZH(NSUM)=ZH(NSUM-1)+HL
                FZ(NSUM)=FZ(NSUM-1)
303         CONTINUE
            IG=NSUM
            ZG=ZH(NSUM)
            PSX(NSUM)=0.0
            PSY(NSUM)=0.0
            DO 306 IA =1,NBL(K)
              HL=HBL(K,IA)/NSG(K,IA)
              CALL EAJ(KSH(K),PKE(K),DOB(K,IA),A,B)
              EJ=PEH(K)*B
              DO 306 IN=1,NSG(K,IA)
                H1=ZH(NSUM) -ZG
                H2=H1+HL
C                WRITE(*,'(3F10.4)') H1,H2,ZBL
                DO 304 I=1,4
                  XA(I)=FX(NSUM,I)
304               XC(I)=FY(NSUM,I)
                XA(4)=-XA(4)
                XC(2)=-XC(2)
                CALL SAA(BTX(K,IA),EJ,H1,H2,R)
                CALL MULULT(4,4,1,R,XA,XB)
                IF(ABS(BTX(K,IA)-BTY(K,IA)).GT.1.0E-3) 
     !              CALL SAA(BTY(K,IA),EJ,H1,H2,R)
                CALL MULULT(4,4,1,R,XC,XD)
                NSUM=NSUM+1
                DO 305 I=1,4
                  FX(NSUM,I)=XB(I)
305               FY(NSUM,I)=XD(I)
                FX(NSUM,4)=-XB(4)
                FY(NSUM,2)=-XD(2)
                ZH(NSUM)=ZH(NSUM-1)+HL
                PSX(NSUM)=FX(NSUM,1)*H2*PMT(K,IA)
                PSY(NSUM)=FY(NSUM,1)*H2*PMT(K,IA)
                IF(KSU(K).GE.3) THEN
                  FZ(NSUM)=FZ(NSUM-1)
                ELSE
                  FZ(NSUM)=FZ(IG)*(1.0-H2**2/ZBL**2)
                END IF
306         CONTINUE
         WRITE(52,820)
         WRITE(52,821) K
         WRITE(52,820)
         WRITE(52,822) PXY(K,1),PXY(K,2)
         WRITE(52,824) 
         WRITE(52,'(/15X,3HUX=,E12.4,4H (m),9X,3HNX=,E12.4,4H (t))')
     !     CE(1),PE(1) 
         WRITE(52,'(15X,3HUY=,E12.4,4H (m),9X,3HNY=,E12.4,4H (t))')
     !     CE(2),PE(2) 
         WRITE(52,'(15X,3HUZ=,E12.4,4H (m),9X,3HNZ=,E12.4,4H (t))') 
     !     CE(3),PE(3)
        WRITE(52,'(15X,3HSX=,E12.4,6H (rad),7X,3HMX=,E12.4,6H (t*m))')
     !     CE(4),PE(4) 
        WRITE(52,'(15X,3HSY=,E12.4,6H (rad),7X,3HMY=,E12.4,6H (t*m))')
     !     CE(5),PE(5) 
      WRITE(52,'(15X,3HSZ=,E12.4,6H (rad),7X,3HMZ=,E12.4,6H (t*m)//)')
     !     CE(6),PE(6) 
         WRITE(52,829)
         WRITE(52,825)
         WRITE(52,826) 
         WRITE(52,829)
         WRITE(52,828)
         WRITE(52,831)
         WRITE(52,'(7X,5E14.4)') (ZH(I),FX(I,1),FY(I,1),FY(I,2),
     !       FX(I,2),I=1,IG-1)
         WRITE(52,'(7X,7E14.4)') (ZH(I),FX(I,1),FY(I,1),FY(I,2),
     !       FX(I,2),PSX(I),PSY(I),I=IG,NSUM)
          WRITE(52,'(//)')
          WRITE(52,829)
          WRITE(52,827)
          WRITE(52,829)
          WRITE(52,830)
          WRITE(52,832)
          WRITE(52,'(7X,6E16.4)') (ZH(I),FX(I,3),FY(I,3),FZ(I),FY(I,4),
     !      FX(I,4),I=1,NSUM)


c         To output the displacements and forces of the pile body onto 
c         the file [pile.pos].   
          DO 101 I=1,IG-1
          PSX(I)=0.0
          PSY(I)=0.0
 101      CONTINUE


          write(53,'(2i5)') k,nsum
          write(53,'(2e14.4)') pxy(k,1),pxy(k,2)
          write(53,'(6e14.4)') (zh(ij),ij=1,nsum)
          write(53,'(6e14.4)') ((fx(ii,jj),jj=1,4),ii=1,nsum)
          write(53,'(6e14.4)') ((fy(ii,jj),jj=1,4),ii=1,nsum)
          write(53,'(6e14.4)') (fz(ii),ii=1,nsum)
          write(53,'(6e14.4)') (psx(ii),ii=1,nsum)
          write(53,'(6e14.4)') (psy(ii),ii=1,nsum)

          
307       CONTINUE
820     FORMAT(7X,'********************************************************
     !*********************************')
821       FORMAT(34X,'NO. ',I2,' # PILE')
822       FORMAT(/12X,'Coordinator of the pile: (x,y) = (',
     !        E12.4,' ,',E12.4,' )'/)
824    FORMAT(12X,'Displacements and internal forces at the top of pile
     !:')
825      FORMAT(32X,'Displacements of the pile body and')
826      FORMAT(27X,'   Compression stresses of soil (PSX,PSY)')
827      FORMAT(32X,'Internal forces of the pile body')
828      FORMAT(//15X,'Z',12X,'UX',12X,'UY',12X,'SX',12X,'SY',12X,'PSX'
     !          ,12X,'PSY')
829     FORMAT(7X,'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
830      FORMAT(//18X,'Z',14X,'NX',14X,'NY',14X,'NZ',14X,'MX',14X,'MY')
831       FORMAT(14X,'(m)',11X,'(m)',11X,'(m)',10X,'(rad)',9X,'(rad)',9X,
     !   '(t/m2)',9X,'(t/m2)'/)
832       FORMAT(17X,'(m)',12X,'(t)',13X,'(t)',13X,'(t)',12X,'(t*m)',
     !    11X,'(t*m)'/)
          RETURN
          END
          
c*****************************************************************
c         Sub to solve equations with G-S method
c***************************************************************** 
         SUBROUTINE GAOS(N,A,B)
          DIMENSION A(N,N),B(N)
          DO 40 K=1,N
            T=A(K,K)
            DO 10 J=K,N
10            A(K,J)=A(K,J)/T
            B(K)=B(K)/T
            I1=K+1
            DO 40 I=I1,N
              T=A(I,K)
              DO 20 J=I1,N
20              A(I,J)=A(I,J)-T*A(K,J)
              B(I)=B(I)-T*B(K)
40            CONTINUE
          N1=N-1
          DO 60 I1=1,N1
            I=N-I1
            I2=I+1
            T=0.0
            DO 50 J=I2,N
50            T=T+A(I,J)*B(J)
            B(I)=B(I)-T
60        CONTINUE
            RETURN
            END


                


          subroutine f_name(b1,a1,out1)
          character*20 b,b1
          character*24 out,out1
          character*4 a,a1
          character*1 c(20),c1(4),c0(24)
          EQUIVALENCE (b,c)
          EQUIVALENCE (a,c1)
          EQUIVALENCE (out,c0)        
          b=b1
          a=a1

          do 5 i=1,24
            c0(i)=' '
  5       continue

          ii=0

          do 10 i=1,20
           if(c(i).ne.' ') ii=ii+1
 10       continue               
          do 20 i=1,ii
           c0(i)=c(i)
 20       continue
          do 30 i=1,4
           c0(i+ii)=c1(i)
  30      continue
          out1=out


         end
	
