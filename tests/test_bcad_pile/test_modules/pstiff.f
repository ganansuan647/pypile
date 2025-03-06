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
148           KBY(I,J)=0.0
            KBX(I,I)=1.0
149         KBY(I,I)=1.0                
          GOTO  2055
        END IF   
        H(1)=0.0
        DO 150 IA=1,NBL(K)
          BT1(IA)=BTX(K,IA)
          BT2(IA)=BTY(K,IA)
          CALL EAJ(KSH(K),PKE(K),DOB(K,IA),A,B)
          EJ(IA)=PEH(K)*B
150       H(IA+1)=H(IA)+HBL(K,IA)
        CALL RLTMTX(NBL(K),BT1,BT2,EJ,H,KBX,KBY)
2055      IF(NFR(K).EQ.0) THEN
          DO 152 I=1,4
            DO 151 J=1,4
              KX(I,J)=KBX(I,J)
151           KY(I,J)=KBY(I,J)
            KX(I,4)=-KX(I,4)
152         KY(I,4)=-KY(I,4)
          GOTO 2060
        END IF
        DO 153 IA=1,NFR(K)
          CALL EAJ(KSH(K),PKE(K),DOF(K,IA),A,B)
          EJ(IA)=PEH(K)*B
153       H(IA)=HFR(K,IA)
        CALL RLTFR(NFR(K),EJ,H,KFR)
        CALL COMBX(KBX,KFR,KX)
        CALL COMBX(KBY,KFR,KY)
2060    CALL CNDTN(KSU(K),KX,KY,RZZ(K),KE)
        DO 154 I=1,6
          DO 154 J=1,6
            K1=(K-1)*6
154         ESP(K1+I,J)=KE(I,J)
155   CONTINUE
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
      CALL SAA(BT1(1),EJ(1),H(1),H(2),KBX)
      DO 161 IA=2,NBL
        DO 160 I1=1,4
          DO 160 J1=1,4
160         A1(I1,J1)=KBX(I1,J1)
        CALL SAA(BT1(IA),EJ(IA),H(IA),H(IA+1),A2)
        CALL MULULT(4,4,4,A2,A1,KBX)
161   CONTINUE  
      DO 162 IA=1,NBL
        IF(ABS(BT2(IA)-BT1(IA)).GT.1.0E-10) GOTO 2300
162   CONTINUE
      DO 163 I1=1,4
        DO 163 J1=1,4
163       KBY(I1,J1)=KBX(I1,J1)
      GOTO 2400
2300  CALL SAA(BT2(1),EJ(1),H(1),H(2),KBY)
      DO 165 IA=2,NBL
        DO 164 I1=1,4
          DO 164 J1=1,4
164         A1(I1,J1)=KBY(I1,J1)
        CALL SAA(BT2(IA),EJ(IA),H(IA),H(IA+1),A2)
        CALL MULULT(4,4,4,A2,A1,KBY)
165   CONTINUE
2400  RETURN
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
      CALL MULULT(4,4,4,AI2,AI3,AI)
      DO 167 I=1,2
        DO 167 J=1,2
          AI(I,J+2)=AI(I,J+2)/EJ
167       AI(I+2,J)=AI(I+2,J)*EJ
      DO 168 J=1,4
        X=AI(3,J)
        AI(3,J)=AI(4,J)
168     AI(4,J)=X
      DO 169 I=1,4
        X=AI(I,3)
        AI(I,3)=AI(I,4)
169     AI(I,4)=X
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
10        C(I,K)=C(I,K)+A(I,J)*B(J,K)
      RETURN
      END
