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
          
c*****************************************************************
c Sub to calculate the multiplication of two matrices.
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