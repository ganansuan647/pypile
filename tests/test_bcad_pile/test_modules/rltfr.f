c*************************************************************
c  Sub to calculate relational matrix of free pile segments 
c*************************************************************
c
      SUBROUTINE RLTFR(N,EJ,H,KFR)
      DIMENSION EJ(N),H(N),KFR(4,4),KF(4,4),KF1(4,4),KF2(4,4)
      REAL KFR,KF,KF1,KF2
      CALL FRLMTX(EJ(1),H(1),KF)
      IF(N.EQ.1) THEN
        DO 186 I=1,4
          DO 186 J=1,4
186         KFR(I,J)=KF(I,J)
        GOTO 2200
      END IF
      DO 187 I=1,4
        DO 187 J=1,4
187       KF1(I,J)=KF(I,J)
      DO 188 I=2,N
        CALL FRLMTX(EJ(I),H(I),KF2)
        CALL FRFRCOM(KF1,KF2,KF)
        DO 188 I1=1,4
          DO 188 J1=1,4
188         KF1(I1,J1)=KF(I1,J1)
      DO 189 I=1,4
        DO 189 J=1,4
189       KFR(I,J)=KF1(I,J)
2200  RETURN
      END

c***********************************************************
c   Sub to calculate relational matrix of a free segment
c***********************************************************
c
      SUBROUTINE FRLMTX(EJ,H,KF)
      REAL KF(4,4)
      X=H/EJ
      KF(1,1)=12.0/X
      KF(1,2)=6.0
      KF(2,1)=6.0
      KF(2,2)=4.0*H
      KF(3,3)=12.0/X
      KF(3,4)=6.0
      KF(4,3)=6.0
      KF(4,4)=4.0*H
      KF(1,3)=-12.0/X
      KF(1,4)=6.0
      KF(2,3)=-6.0
      KF(2,4)=2.0*H
      KF(3,1)=KF(1,3)
      KF(3,2)=KF(2,3)
      KF(4,1)=KF(1,4)
      KF(4,2)=KF(2,4)
      RETURN
      END

c***************************************************************
c  Sub to combine free pile segments 
c***************************************************************
c
      SUBROUTINE FRFRCOM(KF1,KF2,KF)
      REAL KF1(4,4),KF2(4,4),KF(4,4),KF3(4,4),KF4(4,4),KF5(4,4)
      DIMENSION AX(2,2),BX(2,2),CX(2,2),DX(2,2),X(2,2),Y(2,2),Z(2,2)
      DO 400 I=1,2
        DO 400 J=1,2
          I2=I+2
          J2=J+2
          AX(I,J)=KF1(I,J)
          BX(I,J)=KF1(I,J2)
          CX(I,J)=KF1(I2,J)
          DX(I,J)=KF1(I2,J2)+KF2(I,J)
400   CONTINUE
      CALL SINVER(DX,2,X,JE)
      DO 401 I=1,2
        DO 401 J=1,2
401       Y(I,J)=CX(I,J)*X(J,1)+CX(I,2)*X(2,J)
      DO 402 I=1,2
        DO 402 J=1,2
          Z(I,J)=0.0
          DO 402 K=1,2
402         Z(I,J)=Z(I,J)+Y(I,K)*BX(K,J)
      DO 403 I=1,2
        DO 403 J=1,2
          I2=I+2
          J2=J+2
          KF(I,J)=AX(I,J)-Z(I,J)
          KF(I,J2)=BX(I,J)*X(J,1)+BX(I,2)*X(2,J)
          KF(I2,J)=KF2(I,J)*X(J,1)+KF2(I,2)*X(2,J)
403   CONTINUE
      DO 404 I=1,2
        DO 404 J=1,2
          I2=I+2
          J2=J+2
          KF(I2,J2)=KF2(I2,J2)
          Z(I,J)=BX(1,I)*X(1,J)+BX(2,I)*X(2,J)
404   CONTINUE
      DO 405 I=1,2
        DO 405 J=1,2
          KF3(I,J)=0.0
          DO 405 K=1,2
405         KF3(I,J)=KF3(I,J)+Z(I,K)*KF2(K,J)
      DO 406 I=1,2
        DO 406 J=1,2
          KF4(I,J)=0.0
          DO 406 K=1,2
406         KF4(I,J)=KF4(I,J)+KF2(I,K)*Z(K,J)
      DO 407 I=1,2
        DO 407 J=1,2
          KF5(I,J)=0.0
          DO 407 K=1,2
407         KF5(I,J)=KF5(I,J)+KF2(I,K)*X(K,J)
      DO 408 I=1,2
        DO 408 J=1,2
          I2=I+2
          J2=J+2
          KF(I2,J2)=KF(I2,J2)-KF5(I,J2)*KF2(J,I2)
     +                        +KF4(I,J)+KF3(I,J)
408   CONTINUE
      RETURN
      END
