c*************************************************************
c  Sub to calculate relational matrix of free pile segments 
c*************************************************************
c
      SUBROUTINE RLTFR(N,EJ,H,KFR)
      INTEGER N
      REAL EJ(N),H(N),KFR(4,4),KF(4,4),KF1(4,4),KF2(4,4)
      CALL FRLMTX(EJ(1),H(1),KF)
      IF(N.EQ.1) THEN
        DO I=1,4
          DO J=1,4
            KFR(I,J)=KF(I,J)
          END DO
        END DO
        GOTO 2200
      END IF
      DO I=1,4
        DO J=1,4
          KF1(I,J)=KF(I,J)
        END DO
      END DO
      DO I=2,N
        CALL FRLMTX(EJ(I),H(I),KF2)
        CALL FRFRCOM(KF1,KF2,KF)
        DO I1=1,4
          DO J1=1,4
            KF1(I1,J1)=KF(I1,J1)
          END DO
        END DO
      END DO
      DO I=1,4
        DO J=1,4
          KFR(I,J)=KF1(I,J)
        END DO
      END DO
2200  RETURN
      END

c***********************************************************
c   Sub to calculate relational matrix of a free segment
c***********************************************************
c
      SUBROUTINE FRLMTX(EJ,H,KF)
      REAL EJ,H,KF(4,4)
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
      REAL KF1(4,4),KF2(4,4),KF(4,4),KF3(2,2),KF4(2,2),KF5(2,2)
      DIMENSION AX(2,2),BX(2,2),CX(2,2),DX(2,2),X(2,2),Y(2,2),Z(2,2)
      
      ! Initialize KF3, KF4, and KF5 arrays to zero
      DO I=1,2
        DO J=1,2
          KF3(I,J)=0.0
          KF4(I,J)=0.0
          KF5(I,J)=0.0
        END DO
      END DO
      
      DO I=1,2
        DO J=1,2
          I2=I+2
          J2=J+2
          AX(I,J)=KF1(I,J)
          BX(I,J)=KF1(I,J2)
          CX(I,J)=KF1(I2,J)
          DX(I,J)=KF1(I2,J2)+KF2(I,J)
        END DO
      END DO
      
      CALL SINVER(DX,2,X,JE)
      
      DO I=1,2
        DO J=1,2
          Y(I,J)=CX(I,1)*X(1,J)+CX(I,2)*X(2,J)
        END DO
      END DO
      
      DO I=1,2
        DO J=1,2
          Z(I,J)=0.0
          DO K=1,2
            Z(I,J)=Z(I,J)+Y(I,K)*BX(K,J)
          END DO
        END DO
      END DO
      
      DO I=1,2
        DO J=1,2
          I2=I+2
          J2=J+2
          KF(I,J)=AX(I,J)-Z(I,J)
          KF(I,J2)=BX(I,1)*X(1,J)+BX(I,2)*X(2,J)
          KF(I2,J)=KF2(I,1)*X(1,J)+KF2(I,2)*X(2,J)
        END DO
      END DO
      
      DO I=1,2
        DO J=1,2
          I2=I+2
          J2=J+2
          KF(I2,J2)=KF2(I2,J2)
          Z(I,J)=BX(1,I)*X(1,J)+BX(2,I)*X(2,J)
        END DO
      END DO
      
      DO I=1,2
        DO J=1,2
          KF3(I,J)=0.0
          DO K=1,2
            KF3(I,J)=KF3(I,J)+Z(I,K)*KF2(K,J)
          END DO
        END DO
      END DO
      
      DO I=1,2
        DO J=1,2
          KF4(I,J)=0.0
          DO K=1,2
            KF4(I,J)=KF4(I,J)+KF2(I,K)*Z(K,J)
          END DO
        END DO
      END DO
      
      DO I=1,2
        DO J=1,2
          KF5(I,J)=0.0
          DO K=1,2
            KF5(I,J)=KF5(I,J)+KF2(I,K)*X(K,J)
          END DO
        END DO
      END DO
      
      DO I=1,2
        DO J=1,2
          I2=I+2
          J2=J+2
          KF(I2,J2)=KF(I2,J2)-KF5(I,J)*KF2(J,I2)+KF4(I,J)+KF3(I,J)
        END DO
      END DO
      
      RETURN
      END

c***************************************************************
c  Sub to calculate matrix inversion
c***************************************************************
c
      SUBROUTINE SINVER(A,N,B,ERROR)
      INTEGER N,ERROR
      REAL A(N,N),B(N,N)
      INTEGER IS(10),JS(10)
      
      ! Initialize B as identity matrix
      DO I=1,N
        DO J=1,N
          IF(I.EQ.J) THEN
            B(I,J)=1.0
          ELSE
            B(I,J)=0.0
          END IF
        END DO
      END DO
      
      ! Initialize error code
      ERROR=0
      
      ! Gaussian elimination with partial pivoting
      DO K=1,N
        ! Find pivot element
        D=0.0
        DO I=K,N
          DO J=K,N
            IF(ABS(A(I,J)).GT.D) THEN
              D=ABS(A(I,J))
              IS(K)=I
              JS(K)=J
            END IF
          END DO
        END DO
        
        ! Check if matrix is singular
        IF(D.EQ.0.0) THEN
          ERROR=1
          RETURN
        END IF
        
        ! Swap rows and columns
        DO J=1,N
          ! Swap rows
          TEMP=A(IS(K),J)
          A(IS(K),J)=A(K,J)
          A(K,J)=TEMP
          
          ! Swap columns
          TEMP=B(J,IS(K))
          B(J,IS(K))=B(J,K)
          B(J,K)=TEMP
        END DO
        
        DO I=1,N
          ! Swap columns
          TEMP=A(I,JS(K))
          A(I,JS(K))=A(I,K)
          A(I,K)=TEMP
          
          ! Swap rows
          TEMP=B(JS(K),I)
          B(JS(K),I)=B(K,I)
          B(K,I)=TEMP
        END DO
        
        ! Pivot
        A(K,K)=1.0/A(K,K)
        DO J=1,N
          IF(J.NE.K) THEN
            A(K,J)=A(K,J)*A(K,K)
            B(K,J)=B(K,J)*A(K,K)
          END IF
        END DO
        
        DO I=1,N
          IF(I.NE.K) THEN
            DO J=1,N
              IF(J.NE.K) THEN
                A(I,J)=A(I,J)-A(I,K)*A(K,J)
                B(I,J)=B(I,J)-A(I,K)*B(K,J)
              END IF
            END DO
          END IF
        END DO
        
        DO I=1,N
          IF(I.NE.K) THEN
            A(I,K)=-A(I,K)*A(K,K)
            B(I,K)=-A(I,K)*B(K,K)
          END IF
        END DO
      END DO
      
      ! Restore order
      DO K=N,1,-1
        DO J=1,N
          TEMP=B(JS(K),J)
          B(JS(K),J)=B(K,J)
          B(K,J)=TEMP
        END DO
        
        DO I=1,N
          TEMP=B(I,IS(K))
          B(I,IS(K))=B(I,K)
          B(I,K)=TEMP
        END DO
      END DO
      
      RETURN
      END
