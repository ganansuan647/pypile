c*******************************************************************
c   Sub to inverse a matrix
c*******************************************************************
c
      SUBROUTINE SINVER(A,N,B,JE)
      DIMENSION A(N,N),B(N,N),IS(10),JS(10)
      JE=0
      DO 10 I=1,N
      DO 10 J=1,N
10    B(I,J)=0.0
      DO 15 I=1,N
15    B(I,I)=1.0
      DO 100 K=1,N
      D=0.0
      DO 30 I=K,N
      DO 30 J=K,N
      IF(ABS(A(I,J)).LE.ABS(D)) GOTO 30
      D=A(I,J)
      IS(K)=I
      JS(K)=J
30    CONTINUE
      IF(ABS(D).LT.1.0E-10) THEN
        JE=1
        RETURN
      END IF
      DO 40 J=1,N
      C=A(K,J)
      A(K,J)=A(IS(K),J)
40    A(IS(K),J)=C
      DO 50 I=1,N
      C=A(I,K)
      A(I,K)=A(I,JS(K))
50    A(I,JS(K))=C
      DO 60 I=1,N
      IF(I.EQ.K) GOTO 60
      A(I,K)=-A(I,K)/D
60    CONTINUE
      DO 80 I=1,N
      IF(I.EQ.K) GOTO 80
      DO 70 J=1,N
      IF(J.EQ.K) GOTO 70
      A(I,J)=A(I,J)+A(I,K)*A(K,J)
70    CONTINUE
80    CONTINUE
      DO 90 J=1,N
      IF(J.EQ.K) GOTO 90
      A(K,J)=A(K,J)/D
90    CONTINUE
      A(K,K)=1.0/D
100   CONTINUE
      DO 130 L=1,N
      K=N-L+1
      DO 110 J=1,N
      C=B(K,J)
      B(K,J)=B(IS(K),J)
110   B(IS(K),J)=C
      DO 120 I=1,N
      C=B(I,K)
      B(I,K)=B(I,JS(K))
120   B(I,JS(K))=C
130   CONTINUE
      RETURN
      END
