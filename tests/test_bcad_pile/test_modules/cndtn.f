c*****************************************************************
c    Sub to deal with boundary conditions
c*****************************************************************
c
      SUBROUTINE CNDTN(KSU,KX,KY,RZZ,KE)
      REAL KX(4,4),KY(4,4),KE(6,6)
      DO 170 I=1,6
        DO 170 J=1,6
170     KE(I,J)=0.0
      KE(1,1)=KX(2,2)
      KE(1,5)=-KX(2,3)
      KE(2,2)=KY(2,2)
      KE(2,4)=KY(2,3)
      KE(3,3)=RZZ
      KE(4,2)=KY(3,2)
      KE(4,4)=KY(3,3)
      KE(5,1)=-KX(3,2)
      KE(5,5)=KX(3,3)
      KE(6,6)=KX(4,4)+KY(4,4)
      IF(KSU.EQ.2) THEN
        KE(2,6)=KY(2,4)+KE(2,5)
        KE(4,6)=KY(3,4)+KE(4,5)
        KE(6,2)=KY(4,2)+KE(6,1)
        KE(6,4)=KY(4,3)+KE(6,3)
      ELSE
        KE(1,6)=KX(2,4)
        KE(2,6)=KY(2,4)
        KE(4,6)=KY(3,4)
        KE(5,6)=KX(3,4)
        KE(6,1)=KX(4,2)
        KE(6,2)=KY(4,2)
        KE(6,4)=KY(4,3)
        KE(6,5)=KX(4,3)
      END IF
      DO 173 I=1,5
        DO 173 J=I+1,6
173       KE(J,I)=KE(I,J)
      IF(KSU.EQ.4) KE(3,3)=100.0*KE(3,3)
      RETURN
      END
