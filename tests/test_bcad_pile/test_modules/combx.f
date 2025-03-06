c*****************************************************************
c    Sub to combine rigid matrix of pile foundation
c*****************************************************************
c
      SUBROUTINE COMBX(KBX,KFR,K1)
      REAL KBX(4,4),KFR(4,4),K1(4,4)
      DO 171 I=1,4
        DO 171 J=1,4
171     K1(I,J)=0.0
      DO 172 I=1,2
        DO 172 J=1,2
172       K1(I,J)=KBX(I,J)+KFR(1,1)
      K1(1,2)=K1(1,2)-KFR(1,2)
      K1(1,3)=KBX(1,3)+KFR(1,3)
      K1(1,4)=KBX(1,4)
      K1(2,1)=K1(2,1)-KFR(2,1)
      K1(2,2)=K1(2,2)+KFR(2,2)
      K1(2,3)=KBX(2,3)-KFR(2,3)
      K1(2,4)=KBX(2,4)
      K1(3,1)=KBX(3,1)+KFR(3,1)
      K1(3,2)=KBX(3,2)-KFR(3,2)
      K1(3,3)=KBX(3,3)+KFR(3,3)
      K1(3,4)=KBX(3,4)
      K1(4,1)=KBX(4,1)
      K1(4,2)=KBX(4,2)
      K1(4,3)=KBX(4,3)
      K1(4,4)=KBX(4,4)
      RETURN
      END
