c*****************************************************************
c     Sub to calculate displacements at the cap of a pile foundation
c*****************************************************************
c
      SUBROUTINE DISP(JCTR,INO,PNUM,SNUM,PXY,SXY,AGL,FORCE,DUK,SO)
      COMMON /ESTIF/ ESP(1000000,6)
      INTEGER PNUM,SNUM
      DIMENSION KP(1000),KS(1000),PXY(1000,2),SXY(1000,2),
     !      AGL(1000,3),FORCE(6),DUK(100),SO(1000000)
      REAL KE(6,6),DU(6,1),FR(6,1)
      IF(INO.GT.1) GOTO 250
      PES=0.0
        DO 215 I=1,PNUM
          I1=(I-1)*6
          DO 215 J=1,6
215         DUK(I1+J)=0.0
250   DO 217 I=1,PNUM
217     KP(I)=0
      DO 218 I=1,SNUM
218     KS(I)=0
      DO 219 I=1,6
        FR(I,1)=FORCE(I)
219     DU(I,1)=0.0
      IF(JCTR.EQ.1) THEN
        GOTO 221
      ELSE
        GOTO 225
      END IF
221   DO 222 I=1,PNUM
        K1=6*(I-1)
        KP(I)=K1
        DO 222 J=1,6
222       DUK(K1+J)=0.0
      GOTO 230
225   DO 226 K1=1,PNUM
        K0=(K1-1)*6
        R=-PXY(K1,1)*FORCE(5)+PXY(K1,2)*FORCE(4)
        DUK(K0+1)=0.001*FORCE(1)/PNUM
        DUK(K0+2)=0.001*FORCE(2)/PNUM
        DUK(K0+3)=0.001*FORCE(3)/PNUM
        DUK(K0+4)=0.001*FORCE(4)
        DUK(K0+5)=0.001*FORCE(5)
        DUK(K0+6)=0.001*FORCE(6)
        IF(AGL(K1,3).EQ.1.0) GOTO 226 
        ASG=SQRT(1-AGL(K1,3)**2)
        ASG1=AGL(K1,1)/ASG
        ASG2=AGL(K1,2)/ASG
        DUK(K0+1)=DUK(K0+1)+0.001*ASG1*R
        DUK(K0+2)=DUK(K0+2)+0.001*ASG2*R
        DUK(K0+3)=DUK(K0+3)-0.001*ASG*R
226   CONTINUE
230   DO 228 I=1,SNUM
228     KS(I)=PNUM*6+(I-1)*6
      DO 231 I1=1,6
        DO 231 I2=1,6
231       KE(I1,I2)=0.0
      K1=0
      DO 235 K=1,PNUM
        K00=KP(K)
        ESI=0.0
        DO 234 I=1,6
          DO 234 J=1,6
            K00=KP(K)
            K01=(K-1)*6
            KE(I,J)=ESP(K01+I,J)
            SO(K1+1)=KE(I,J)
234         K1=K1+1
        DO 235 I=1,6
          DO 235 J=1,6
            FR(I,1)=FR(I,1)-KE(I,J)*DUK(K00+J)
235         ESI=ESI+KE(I,J)*DUK(K00+J)**2
      PES=PES+ESI
      K1=0
      DO 239 K=1,SNUM
        DO 238 I=1,6
          DO 238 J=1,6
            K3=PNUM*6*6+(K-1)*6*6
            SO(K3+K1+1)=0.0
238         K1=K1+1
239   CONTINUE
      RETURN
      END
