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
140   CONTINUE
      LO=100.0
      DO 141 I=1,IN
        DO 141 I1=I+1,IN
          S=ABS(AA(I)-AA(I1))-(DD(I)+DD(I1))/2.0
          IF(S.LT.LO) THEN
            LO=S
            HOO=HO(I)
            IF(HOO.LT.HO(I1)) HOO=HO(I1)
          END IF
141   CONTINUE  
      IF(LO.GE.0.6*HOO) THEN
        KINF=1.0
      ELSE
        CALL PARC(IN,C)
        KINF=C+(1.0-C)*LO/(0.6*HOO)
      END IF
2200  RETURN
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
