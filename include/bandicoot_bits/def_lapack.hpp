// Copyright 2017 Conrad Sanderson (http://conradsanderson.id.au)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------



// TODO: remove functions that are actually not used

#if !defined(COOT_BLAS_CAPITALS)

  #define coot_sgetrf sgetrf
  #define coot_dgetrf dgetrf
  #define coot_cgetrf cgetrf
  #define coot_zgetrf zgetrf

  #define coot_sgetri sgetri
  #define coot_dgetri dgetri
  #define coot_cgetri cgetri
  #define coot_zgetri zgetri

  #define coot_strtri strtri
  #define coot_dtrtri dtrtri
  #define coot_ctrtri ctrtri
  #define coot_ztrtri ztrtri

  #define coot_sgeev  sgeev
  #define coot_dgeev  dgeev
  #define coot_cgeev  cgeev
  #define coot_zgeev  zgeev

  #define coot_ssyev  ssyev
  #define coot_dsyev  dsyev

  #define coot_cheev  cheev
  #define coot_zheev  zheev

  #define coot_ssyevd ssyevd
  #define coot_dsyevd dsyevd

  #define coot_cheevd cheevd
  #define coot_zheevd zheevd

  #define coot_sggev  sggev
  #define coot_dggev  dggev

  #define coot_cggev  cggev
  #define coot_zggev  zggev

  #define coot_spotrf spotrf
  #define coot_dpotrf dpotrf
  #define coot_cpotrf cpotrf
  #define coot_zpotrf zpotrf

  #define coot_spotri spotri
  #define coot_dpotri dpotri
  #define coot_cpotri cpotri
  #define coot_zpotri zpotri

  #define coot_sgeqrf sgeqrf
  #define coot_dgeqrf dgeqrf
  #define coot_cgeqrf cgeqrf
  #define coot_zgeqrf zgeqrf

  #define coot_sorgqr sorgqr
  #define coot_dorgqr dorgqr

  #define coot_cungqr cungqr
  #define coot_zungqr zungqr

  #define coot_sgesvd sgesvd
  #define coot_dgesvd dgesvd

  #define coot_cgesvd cgesvd
  #define coot_zgesvd zgesvd

  #define coot_sgesdd sgesdd
  #define coot_dgesdd dgesdd
  #define coot_cgesdd cgesdd
  #define coot_zgesdd zgesdd

  #define coot_sgesv  sgesv
  #define coot_dgesv  dgesv
  #define coot_cgesv  cgesv
  #define coot_zgesv  zgesv

  #define coot_sgesvx sgesvx
  #define coot_dgesvx dgesvx
  #define coot_cgesvx cgesvx
  #define coot_zgesvx zgesvx

  #define coot_sgels  sgels
  #define coot_dgels  dgels
  #define coot_cgels  cgels
  #define coot_zgels  zgels

  #define coot_sgelsd sgelsd
  #define coot_dgelsd dgelsd
  #define coot_cgelsd cgelsd
  #define coot_zgelsd zgelsd

  #define coot_strtrs strtrs
  #define coot_dtrtrs dtrtrs
  #define coot_ctrtrs ctrtrs
  #define coot_ztrtrs ztrtrs

  #define coot_sgees  sgees
  #define coot_dgees  dgees
  #define coot_cgees  cgees
  #define coot_zgees  zgees

  #define coot_strsyl strsyl
  #define coot_dtrsyl dtrsyl
  #define coot_ctrsyl ctrsyl
  #define coot_ztrsyl ztrsyl

  #define coot_ssytrf ssytrf
  #define coot_dsytrf dsytrf
  #define coot_csytrf csytrf
  #define coot_zsytrf zsytrf

  #define coot_ssytri ssytri
  #define coot_dsytri dsytri
  #define coot_csytri csytri
  #define coot_zsytri zsytri

  #define coot_sgges  sgges
  #define coot_dgges  dgges
  #define coot_cgges  cgges
  #define coot_zgges  zgges

  #define coot_slange slange
  #define coot_dlange dlange
  #define coot_clange clange
  #define coot_zlange zlange

  #define coot_slansy slansy
  #define coot_dlansy dlansy
  #define coot_clansy clansy
  #define coot_zlansy zlansy

  #define coot_sgecon sgecon
  #define coot_dgecon dgecon
  #define coot_cgecon cgecon
  #define coot_zgecon zgecon

  #define coot_ilaenv ilaenv

  #define coot_ssytrs ssytrs
  #define coot_dsytrs dsytrs
  #define coot_csytrs csytrs
  #define coot_zsytrs zsytrs

  #define coot_sgetrs sgetrs
  #define coot_dgetrs dgetrs
  #define coot_cgetrs cgetrs
  #define coot_zgetrs zgetrs

  #define coot_slahqr slahqr
  #define coot_dlahqr dlahqr

  #define coot_sstedc sstedc
  #define coot_dstedc dstedc

  #define coot_strevc strevc
  #define coot_dtrevc dtrevc

  #define coot_slarnv slarnv
  #define coot_dlarnv dlarnv

  #define coot_slarft slarft
  #define coot_dlarft dlarft
  #define coot_clarft clarft
  #define coot_zlarft zlarft

  #define coot_slarfg slarfg
  #define coot_dlarfg dlarfg
  #define coot_clarfg clarfg
  #define coot_zlarfg zlarfg

  #define coot_sgebrd sgebrd
  #define coot_dgebrd dgebrd
  #define coot_cgebrd cgebrd
  #define coot_zgebrd zgebrd

  #define coot_sorgbr sorgbr
  #define coot_dorgbr dorgbr

  #define coot_sorglq sorglq
  #define coot_dorglq dorglq

  #define coot_sormqr sormqr
  #define coot_dormqr dormqr

  #define coot_sormlq sormlq
  #define coot_dormlq dormlq

  #define coot_sormbr sormbr
  #define coot_dormbr dormbr

  #define coot_slacpy slacpy
  #define coot_dlacpy dlacpy
  #define coot_clacpy clacpy
  #define coot_zlacpy zlacpy

  #define coot_slaset slaset
  #define coot_dlaset dlaset
  #define coot_claset claset
  #define coot_zlaset zlaset

  #define coot_slarfb slarfb
  #define coot_dlarfb dlarfb
  #define coot_clarfb clarfb
  #define coot_zlarfb zlarfb

  #define coot_sgelqf sgelqf
  #define coot_dgelqf dgelqf
  #define coot_cgelqf cgelqf
  #define coot_zgelqf zgelqf

  #define coot_slamch slamch
  #define coot_dlamch dlamch

  #define coot_slascl slascl
  #define coot_dlascl dlascl
  #define coot_clascl clascl
  #define coot_zlascl zlascl

  #define coot_sbdsqr sbdsqr
  #define coot_dbdsqr dbdsqr
  #define coot_cbdsqr cbdsqr
  #define coot_zbdsqr zbdsqr

  #define coot_slaed2 slaed2
  #define coot_dlaed2 dlaed2

  #define coot_ssteqr ssteqr
  #define coot_dsteqr dsteqr
  #define coot_csteqr csteqr
  #define coot_zsteqr zsteqr

  #define coot_slanst slanst
  #define coot_dlanst dlanst

  #define coot_ssytrd ssytrd
  #define coot_dsytrd dsytrd

  #define coot_slamc3 slamc3
  #define coot_dlamc3 dlamc3

  #define coot_slaed4 slaed4
  #define coot_dlaed4 dlaed4

  #define coot_slamrg slamrg
  #define coot_dlamrg dlamrg

  #define coot_sorgtr sorgtr
  #define coot_dorgtr dorgtr

#else

  #define coot_sgetrf SGETRF
  #define coot_dgetrf DGETRF
  #define coot_cgetrf CGETRF
  #define coot_zgetrf ZGETRF

  #define coot_sgetri SGETRI
  #define coot_dgetri DGETRI
  #define coot_cgetri CGETRI
  #define coot_zgetri ZGETRI

  #define coot_strtri STRTRI
  #define coot_dtrtri DTRTRI
  #define coot_ctrtri CTRTRI
  #define coot_ztrtri ZTRTRI

  #define coot_sgeev  SGEEV
  #define coot_dgeev  DGEEV
  #define coot_cgeev  CGEEV
  #define coot_zgeev  ZGEEV

  #define coot_ssyev  SSYEV
  #define coot_dsyev  DSYEV

  #define coot_cheev  CHEEV
  #define coot_zheev  ZHEEV

  #define coot_ssyevd SSYEVD
  #define coot_dsyevd DSYEVD

  #define coot_cheevd CHEEVD
  #define coot_zheevd ZHEEVD

  #define coot_sggev  SGGEV
  #define coot_dggev  DGGEV

  #define coot_cggev  CGGEV
  #define coot_zggev  ZGGEV

  #define coot_spotrf SPOTRF
  #define coot_dpotrf DPOTRF
  #define coot_cpotrf CPOTRF
  #define coot_zpotrf ZPOTRF

  #define coot_spotri SPOTRI
  #define coot_dpotri DPOTRI
  #define coot_cpotri CPOTRI
  #define coot_zpotri ZPOTRI

  #define coot_sgeqrf SGEQRF
  #define coot_dgeqrf DGEQRF
  #define coot_cgeqrf CGEQRF
  #define coot_zgeqrf ZGEQRF

  #define coot_sorgqr SORGQR
  #define coot_dorgqr DORGQR

  #define coot_cungqr CUNGQR
  #define coot_zungqr ZUNGQR

  #define coot_sgesvd SGESVD
  #define coot_dgesvd DGESVD

  #define coot_cgesvd CGESVD
  #define coot_zgesvd ZGESVD

  #define coot_sgesdd SGESDD
  #define coot_dgesdd DGESDD
  #define coot_cgesdd CGESDD
  #define coot_zgesdd ZGESDD

  #define coot_sgesv  SGESV
  #define coot_dgesv  DGESV
  #define coot_cgesv  CGESV
  #define coot_zgesv  ZGESV

  #define coot_sgesvx SGESVX
  #define coot_dgesvx DGESVX
  #define coot_cgesvx CGESVX
  #define coot_zgesvx ZGESVX

  #define coot_sgels  SGELS
  #define coot_dgels  DGELS
  #define coot_cgels  CGELS
  #define coot_zgels  ZGELS

  #define coot_sgelsd SGELSD
  #define coot_dgelsd DGELSD
  #define coot_cgelsd CGELSD
  #define coot_zgelsd ZGELSD

  #define coot_strtrs STRTRS
  #define coot_dtrtrs DTRTRS
  #define coot_ctrtrs CTRTRS
  #define coot_ztrtrs ZTRTRS

  #define coot_sgees  SGEES
  #define coot_dgees  DGEES
  #define coot_cgees  CGEES
  #define coot_zgees  ZGEES

  #define coot_strsyl STRSYL
  #define coot_dtrsyl DTRSYL
  #define coot_ctrsyl CTRSYL
  #define coot_ztrsyl ZTRSYL

  #define coot_ssytrf SSYTRF
  #define coot_dsytrf DSYTRF
  #define coot_csytrf CSYTRF
  #define coot_zsytrf ZSYTRF

  #define coot_ssytri SSYTRI
  #define coot_dsytri DSYTRI
  #define coot_csytri CSYTRI
  #define coot_zsytri ZSYTRI

  #define coot_sgges  SGGES
  #define coot_dgges  DGGES
  #define coot_cgges  CGGES
  #define coot_zgges  ZGGES

  #define coot_slange SLANGE
  #define coot_dlange DLANGE
  #define coot_clange CLANGE
  #define coot_zlange ZLANGE

  #define coot_slansy SLANSY
  #define coot_dlansy DLANSY
  #define coot_clansy CLANSY
  #define coot_zlansy ZLANSY

  #define coot_sgecon SGECON
  #define coot_dgecon DGECON
  #define coot_cgecon CGECON
  #define coot_zgecon ZGECON

  #define coot_ilaenv ILAENV

  #define coot_ssytrs SSYTRS
  #define coot_dsytrs DSYTRS
  #define coot_csytrs CSYTRS
  #define coot_zsytrs ZSYTRS

  #define coot_sgetrs SGETRS
  #define coot_dgetrs DGETRS
  #define coot_cgetrs CGETRS
  #define coot_zgetrs ZGETRS

  #define coot_slahqr SLAHQR
  #define coot_dlahqr DLAHQR

  #define coot_sstedc SSTEDC
  #define coot_dstedc DSTEDC

  #define coot_strevc STREVC
  #define coot_dtrevc DTREVC

  #define coot_slarnv SLARNV
  #define coot_dlarnv DLARNV

  #define coot_slarft SLARFT
  #define coot_dlarft DLARFT
  #define coot_clarft CLARFT
  #define coot_zlarft ZLARFT

  #define coot_slarfg SLARFG
  #define coot_dlarfg DLARFG
  #define coot_clarfg CLARFG
  #define coot_zlarfg ZLARFG

  #define coot_sgebrd SGEBRD
  #define coot_dgebrd DGEBRD
  #define coot_cgebrd CGEBRD
  #define coot_zgebrd ZGEBRD

  #define coot_sorgbr SORGBR
  #define coot_dorgbr DORGBR

  #define coot_sorglq SORGLQ
  #define coot_dorglq DORGLQ

  #define coot_sormqr SORMQR
  #define coot_dormqr DORMQR

  #define coot_sormlq SORMLQ
  #define coot_dormlq DORMLQ

  #define coot_sormbr SORMBR
  #define coot_dormbr DORMBR

  #define coot_slacpy SLACPY
  #define coot_dlacpy DLACPY
  #define coot_clacpy CLACPY
  #define coot_zlacpy ZLACPY

  #define coot_slaset SLASET
  #define coot_dlaset DLASET
  #define coot_claset CLASET
  #define coot_zlaset ZLASET

  #define coot_slarfb SLARFB
  #define coot_dlarfb DLARFB
  #define coot_clarfb CLARFB
  #define coot_zlarfb ZLARFB

  #define coot_sgelqf SGELQF
  #define coot_dgelqf DGELQF
  #define coot_cgelqf CGELQF
  #define coot_zgelqf ZGELQF

  #define coot_slamch SLAMCH
  #define coot_dlamch DLAMCH

  #define coot_slascl SLASCL
  #define coot_dlascl DLASCL
  #define coot_clascl CLASCL
  #define coot_zlascl ZLASCL

  #define coot_sbdsqr SBDSQR
  #define coot_dbdsqr DBDSQR
  #define coot_cbdsqr CBDSQR
  #define coot_zbdsqr ZBDSQR

  #define coot_slaed2 SLAED2
  #define coot_dlaed2 DLAED2

  #define coot_ssteqr SSTEQR
  #define coot_dsteqr DSTEQR
  #define coot_csteqr CSTEQR
  #define coot_zsteqr ZSTEQR

  #define coot_slanst SLANST
  #define coot_dlanst DLANST

  #define coot_ssytrd SSYTRD
  #define coot_dsytrd DSYTRD

  #define coot_slamc3 SLAMC3
  #define coot_dlamc3 DLAMC3

  #define coot_slaed4 SLAED4
  #define coot_dlaed4 DLAED4

  #define coot_slamrg SLAMRG
  #define coot_dlamrg DLAMRG

  #define coot_sorgtr SORGTR
  #define coot_dorgtr DORGTR

#endif



extern "C"
  {
  // LU factorisation
  void coot_fortran(coot_sgetrf)(blas_int* m, blas_int* n,  float* a, blas_int* lda, blas_int* ipiv, blas_int* info);
  void coot_fortran(coot_dgetrf)(blas_int* m, blas_int* n, double* a, blas_int* lda, blas_int* ipiv, blas_int* info);
  void coot_fortran(coot_cgetrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda, blas_int* ipiv, blas_int* info);
  void coot_fortran(coot_zgetrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda, blas_int* ipiv, blas_int* info);

  // matrix inversion (using LU factorisation result)
  void coot_fortran(coot_sgetri)(blas_int* n,  float* a, blas_int* lda, blas_int* ipiv,  float* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dgetri)(blas_int* n, double* a, blas_int* lda, blas_int* ipiv, double* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_cgetri)(blas_int* n,  void*  a, blas_int* lda, blas_int* ipiv,   void* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_zgetri)(blas_int* n,  void*  a, blas_int* lda, blas_int* ipiv,   void* work, blas_int* lwork, blas_int* info);

  // matrix inversion (triangular matrices)
  void coot_fortran(coot_strtri)(const char* uplo, const char* diag, blas_int* n,  float* a, blas_int* lda, blas_int* info);
  void coot_fortran(coot_dtrtri)(const char* uplo, const char* diag, blas_int* n, double* a, blas_int* lda, blas_int* info);
  void coot_fortran(coot_ctrtri)(const char* uplo, const char* diag, blas_int* n,   void* a, blas_int* lda, blas_int* info);
  void coot_fortran(coot_ztrtri)(const char* uplo, const char* diag, blas_int* n,   void* a, blas_int* lda, blas_int* info);

  // eigen decomposition of general matrix (real)
  void coot_fortran(coot_sgeev)(const char* jobvl, const char* jobvr, blas_int* N,  float* a, blas_int* lda,  float* wr,  float* wi,  float* vl, blas_int* ldvl,  float* vr, blas_int* ldvr,  float* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dgeev)(const char* jobvl, const char* jobvr, blas_int* N, double* a, blas_int* lda, double* wr, double* wi, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, double* work, blas_int* lwork, blas_int* info);

  // eigen decomposition of general matrix (complex)
  void coot_fortran(coot_cgeev)(const char* jobvl, const char* jobvr, blas_int* N, void* a, blas_int* lda, void* w, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork,  float* rwork, blas_int* info);
  void coot_fortran(coot_zgeev)(const char* jobvl, const char* jobvr, blas_int* N, void* a, blas_int* lda, void* w, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork, double* rwork, blas_int* info);

  // eigen decomposition of symmetric real matrices
  void coot_fortran(coot_ssyev)(const char* jobz, const char* uplo, blas_int* n,  float* a, blas_int* lda,  float* w,  float* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dsyev)(const char* jobz, const char* uplo, blas_int* n, double* a, blas_int* lda, double* w, double* work, blas_int* lwork, blas_int* info);

  // eigen decomposition of hermitian matrices (complex)
  void coot_fortran(coot_cheev)(const char* jobz, const char* uplo, blas_int* n,   void* a, blas_int* lda,  float* w,   void* work, blas_int* lwork,  float* rwork, blas_int* info);
  void coot_fortran(coot_zheev)(const char* jobz, const char* uplo, blas_int* n,   void* a, blas_int* lda, double* w,   void* work, blas_int* lwork, double* rwork, blas_int* info);

  // eigen decomposition of symmetric real matrices by divide and conquer
  void coot_fortran(coot_ssyevd)(const char* jobz, const char* uplo, blas_int* n,  float* a, blas_int* lda,  float* w,  float* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info);
  void coot_fortran(coot_dsyevd)(const char* jobz, const char* uplo, blas_int* n, double* a, blas_int* lda, double* w, double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info);

  // eigen decomposition of hermitian matrices (complex) by divide and conquer
  void coot_fortran(coot_cheevd)(const char* jobz, const char* uplo, blas_int* n,   void* a, blas_int* lda,  float* w,   void* work, blas_int* lwork,  float* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork, blas_int* info);
  void coot_fortran(coot_zheevd)(const char* jobz, const char* uplo, blas_int* n,   void* a, blas_int* lda, double* w,   void* work, blas_int* lwork, double* rwork, blas_int* lrwork, blas_int* iwork, blas_int* liwork, blas_int* info);

  // eigen decomposition of general real matrix pair
  void coot_fortran(coot_sggev)(const char* jobvl, const char* jobvr, blas_int* n,  float* a, blas_int* lda,  float* b, blas_int* ldb,  float* alphar,  float* alphai,  float* beta,  float* vl, blas_int* ldvl,  float* vr, blas_int* ldvr,  float* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dggev)(const char* jobvl, const char* jobvr, blas_int* n, double* a, blas_int* lda, double* b, blas_int* ldb, double* alphar, double* alphai, double* beta, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, double* work, blas_int* lwork, blas_int* info);

  // eigen decomposition of general complex matrix pair
  void coot_fortran(coot_cggev)(const char* jobvl, const char* jobvr, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, void* alpha, void* beta, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork,  float* rwork, blas_int* info);
  void coot_fortran(coot_zggev)(const char* jobvl, const char* jobvr, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, void* alpha, void* beta, void* vl, blas_int* ldvl, void* vr, blas_int* ldvr, void* work, blas_int* lwork, double* rwork, blas_int* info);

  // Cholesky decomposition
  void coot_fortran(coot_spotrf)(const char* uplo, blas_int* n,  float* a, blas_int* lda, blas_int* info);
  void coot_fortran(coot_dpotrf)(const char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* info);
  void coot_fortran(coot_cpotrf)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info);
  void coot_fortran(coot_zpotrf)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info);

  // matrix inversion (using Cholesky decomposition result)
  void coot_fortran(coot_spotri)(const char* uplo, blas_int* n,  float* a, blas_int* lda, blas_int* info);
  void coot_fortran(coot_dpotri)(const char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* info);
  void coot_fortran(coot_cpotri)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info);
  void coot_fortran(coot_zpotri)(const char* uplo, blas_int* n,   void* a, blas_int* lda, blas_int* info);

  // QR decomposition
  void coot_fortran(coot_sgeqrf)(blas_int* m, blas_int* n,  float* a, blas_int* lda,  float* tau,  float* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dgeqrf)(blas_int* m, blas_int* n, double* a, blas_int* lda, double* tau, double* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_cgeqrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_zgeqrf)(blas_int* m, blas_int* n,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info);

  // Q matrix calculation from QR decomposition (real matrices)
  void coot_fortran(coot_sorgqr)(blas_int* m, blas_int* n, blas_int* k,  float* a, blas_int* lda,  float* tau,  float* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dorgqr)(blas_int* m, blas_int* n, blas_int* k, double* a, blas_int* lda, double* tau, double* work, blas_int* lwork, blas_int* info);

  // Q matrix calculation from QR decomposition (complex matrices)
  void coot_fortran(coot_cungqr)(blas_int* m, blas_int* n, blas_int* k,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_zungqr)(blas_int* m, blas_int* n, blas_int* k,   void* a, blas_int* lda,   void* tau,   void* work, blas_int* lwork, blas_int* info);

  // SVD (real matrices)
  void coot_fortran(coot_sgesvd)(const char* jobu, const char* jobvt, blas_int* m, blas_int* n, float*  a, blas_int* lda, float*  s, float*  u, blas_int* ldu, float*  vt, blas_int* ldvt, float*  work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dgesvd)(const char* jobu, const char* jobvt, blas_int* m, blas_int* n, double* a, blas_int* lda, double* s, double* u, blas_int* ldu, double* vt, blas_int* ldvt, double* work, blas_int* lwork, blas_int* info);

  // SVD (complex matrices)
  void coot_fortran(coot_cgesvd)(const char* jobu, const char* jobvt, blas_int* m, blas_int* n, void*   a, blas_int* lda, float*  s, void*   u, blas_int* ldu, void*   vt, blas_int* ldvt, void*   work, blas_int* lwork, float*  rwork, blas_int* info);
  void coot_fortran(coot_zgesvd)(const char* jobu, const char* jobvt, blas_int* m, blas_int* n, void*   a, blas_int* lda, double* s, void*   u, blas_int* ldu, void*   vt, blas_int* ldvt, void*   work, blas_int* lwork, double* rwork, blas_int* info);

  // SVD (real matrices) by divide and conquer
  void coot_fortran(coot_sgesdd)(const char* jobz, blas_int* m, blas_int* n, float*  a, blas_int* lda, float*  s, float*  u, blas_int* ldu, float*  vt, blas_int* ldvt, float*  work, blas_int* lwork, blas_int* iwork, blas_int* info);
  void coot_fortran(coot_dgesdd)(const char* jobz, blas_int* m, blas_int* n, double* a, blas_int* lda, double* s, double* u, blas_int* ldu, double* vt, blas_int* ldvt, double* work, blas_int* lwork, blas_int* iwork, blas_int* info);

  // SVD (complex matrices) by divide and conquer
  void coot_fortran(coot_cgesdd)(const char* jobz, blas_int* m, blas_int* n, void* a, blas_int* lda, float*  s, void* u, blas_int* ldu, void* vt, blas_int* ldvt, void* work, blas_int* lwork, float*  rwork, blas_int* iwork, blas_int* info);
  void coot_fortran(coot_zgesdd)(const char* jobz, blas_int* m, blas_int* n, void* a, blas_int* lda, double* s, void* u, blas_int* ldu, void* vt, blas_int* ldvt, void* work, blas_int* lwork, double* rwork, blas_int* iwork, blas_int* info);

  // solve system of linear equations (general square matrix)
  void coot_fortran(coot_sgesv)(blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, blas_int* ipiv, float*  b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_dgesv)(blas_int* n, blas_int* nrhs, double* a, blas_int* lda, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_cgesv)(blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_zgesv)(blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);

  // solve system of linear equations (general square matrix, advanced form, real matrices)
  void coot_fortran(coot_sgesvx)(const char* fact, const char* trans, blas_int* n, blas_int* nrhs,  float* a, blas_int* lda,  float* af, blas_int* ldaf, blas_int* ipiv, const char* equed,  float* r,  float* c,  float* b, blas_int* ldb,  float* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr,  float* work, blas_int* iwork, blas_int* info);
  void coot_fortran(coot_dgesvx)(const char* fact, const char* trans, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* af, blas_int* ldaf, blas_int* ipiv, const char* equed, double* r, double* c, double* b, blas_int* ldb, double* x, blas_int* ldx, double* rcond, double* ferr, double* berr, double* work, blas_int* iwork, blas_int* info);

  // solve system of linear equations (general square matrix, advanced form, complex matrices)
  void coot_fortran(coot_cgesvx)(const char* fact, const char* trans, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* af, blas_int* ldaf, blas_int* ipiv, const char* equed,  float* r,  float* c, void* b, blas_int* ldb, void* x, blas_int* ldx,  float* rcond,  float* ferr,  float* berr, void* work,  float* rwork, blas_int* info);
  void coot_fortran(coot_zgesvx)(const char* fact, const char* trans, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* af, blas_int* ldaf, blas_int* ipiv, const char* equed, double* r, double* c, void* b, blas_int* ldb, void* x, blas_int* ldx, double* rcond, double* ferr, double* berr, void* work, double* rwork, blas_int* info);

  // solve over/under-determined system of linear equations
  void coot_fortran(coot_sgels)(const char* trans, blas_int* m, blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, float*  b, blas_int* ldb, float*  work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dgels)(const char* trans, blas_int* m, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* b, blas_int* ldb, double* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_cgels)(const char* trans, blas_int* m, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, void*   b, blas_int* ldb, void*   work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_zgels)(const char* trans, blas_int* m, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, void*   b, blas_int* ldb, void*   work, blas_int* lwork, blas_int* info);

  // approximately solve system of linear equations using svd (real)
  void coot_fortran(coot_sgelsd)(blas_int* m, blas_int* n, blas_int* nrhs,  float* a, blas_int* lda,  float* b, blas_int* ldb,  float* S,  float* rcond, blas_int* rank,  float* work, blas_int* lwork, blas_int* iwork, blas_int* info);
  void coot_fortran(coot_dgelsd)(blas_int* m, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, double* b, blas_int* ldb, double* S, double* rcond, blas_int* rank, double* work, blas_int* lwork, blas_int* iwork, blas_int* info);

  // approximately solve system of linear equations using svd (complex)
  void coot_fortran(coot_cgelsd)(blas_int* m, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* b, blas_int* ldb,  float* S,  float* rcond, blas_int* rank, void* work, blas_int* lwork,  float* rwork, blas_int* iwork, blas_int* info);
  void coot_fortran(coot_zgelsd)(blas_int* m, blas_int* n, blas_int* nrhs, void* a, blas_int* lda, void* b, blas_int* ldb, double* S, double* rcond, blas_int* rank, void* work, blas_int* lwork, double* rwork, blas_int* iwork, blas_int* info);

  // solve system of linear equations (triangular matrix)
  void coot_fortran(coot_strtrs)(const char* uplo, const char* trans, const char* diag, blas_int* n, blas_int* nrhs, const float*  a, blas_int* lda, float*  b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_dtrtrs)(const char* uplo, const char* trans, const char* diag, blas_int* n, blas_int* nrhs, const double* a, blas_int* lda, double* b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_ctrtrs)(const char* uplo, const char* trans, const char* diag, blas_int* n, blas_int* nrhs, const void*   a, blas_int* lda, void*   b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_ztrtrs)(const char* uplo, const char* trans, const char* diag, blas_int* n, blas_int* nrhs, const void*   a, blas_int* lda, void*   b, blas_int* ldb, blas_int* info);

  // Schur decomposition (real matrices)
  void coot_fortran(coot_sgees)(const char* jobvs, const char* sort, void* select, blas_int* n, float*  a, blas_int* lda, blas_int* sdim, float*  wr, float*  wi, float*  vs, blas_int* ldvs, float*  work, blas_int* lwork, blas_int* bwork, blas_int* info);
  void coot_fortran(coot_dgees)(const char* jobvs, const char* sort, void* select, blas_int* n, double* a, blas_int* lda, blas_int* sdim, double* wr, double* wi, double* vs, blas_int* ldvs, double* work, blas_int* lwork, blas_int* bwork, blas_int* info);

  // Schur decomposition (complex matrices)
  void coot_fortran(coot_cgees)(const char* jobvs, const char* sort, void* select, blas_int* n, void* a, blas_int* lda, blas_int* sdim, void* w, void* vs, blas_int* ldvs, void* work, blas_int* lwork, float*  rwork, blas_int* bwork, blas_int* info);
  void coot_fortran(coot_zgees)(const char* jobvs, const char* sort, void* select, blas_int* n, void* a, blas_int* lda, blas_int* sdim, void* w, void* vs, blas_int* ldvs, void* work, blas_int* lwork, double* rwork, blas_int* bwork, blas_int* info);

  // solve a Sylvester equation ax + xb = c, with a and b assumed to be in Schur form
  void coot_fortran(coot_strsyl)(const char* transa, const char* transb, blas_int* isgn, blas_int* m, blas_int* n, const float*  a, blas_int* lda, const float*  b, blas_int* ldb, float*  c, blas_int* ldc, float*  scale, blas_int* info);
  void coot_fortran(coot_dtrsyl)(const char* transa, const char* transb, blas_int* isgn, blas_int* m, blas_int* n, const double* a, blas_int* lda, const double* b, blas_int* ldb, double* c, blas_int* ldc, double* scale, blas_int* info);
  void coot_fortran(coot_ctrsyl)(const char* transa, const char* transb, blas_int* isgn, blas_int* m, blas_int* n, const void*   a, blas_int* lda, const void*   b, blas_int* ldb, void*   c, blas_int* ldc, float*  scale, blas_int* info);
  void coot_fortran(coot_ztrsyl)(const char* transa, const char* transb, blas_int* isgn, blas_int* m, blas_int* n, const void*   a, blas_int* lda, const void*   b, blas_int* ldb, void*   c, blas_int* ldc, double* scale, blas_int* info);

  void coot_fortran(coot_ssytrf)(const char* uplo, blas_int* n, float*  a, blas_int* lda, blas_int* ipiv, float*  work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dsytrf)(const char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* ipiv, double* work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_csytrf)(const char* uplo, blas_int* n, void*   a, blas_int* lda, blas_int* ipiv, void*   work, blas_int* lwork, blas_int* info);
  void coot_fortran(coot_zsytrf)(const char* uplo, blas_int* n, void*   a, blas_int* lda, blas_int* ipiv, void*   work, blas_int* lwork, blas_int* info);

  void coot_fortran(coot_ssytri)(const char* uplo, blas_int* n, float*  a, blas_int* lda, blas_int* ipiv, float*  work, blas_int* info);
  void coot_fortran(coot_dsytri)(const char* uplo, blas_int* n, double* a, blas_int* lda, blas_int* ipiv, double* work, blas_int* info);
  void coot_fortran(coot_csytri)(const char* uplo, blas_int* n, void*   a, blas_int* lda, blas_int* ipiv, void*   work, blas_int* info);
  void coot_fortran(coot_zsytri)(const char* uplo, blas_int* n, void*   a, blas_int* lda, blas_int* ipiv, void*   work, blas_int* info);

  // QZ decomposition (real matrices)
  void coot_fortran(coot_sgges)(const char* jobvsl, const char* jobvsr, const char* sort, void* selctg, blas_int* n,  float* a, blas_int* lda,  float* b, blas_int* ldb, blas_int* sdim,  float* alphar,  float* alphai,  float* beta,  float* vsl, blas_int* ldvsl,  float* vsr, blas_int* ldvsr,  float* work, blas_int* lwork,  float* bwork, blas_int* info);
  void coot_fortran(coot_dgges)(const char* jobvsl, const char* jobvsr, const char* sort, void* selctg, blas_int* n, double* a, blas_int* lda, double* b, blas_int* ldb, blas_int* sdim, double* alphar, double* alphai, double* beta, double* vsl, blas_int* ldvsl, double* vsr, blas_int* ldvsr, double* work, blas_int* lwork, double* bwork, blas_int* info);

  // QZ decomposition (complex matrices)
  void coot_fortran(coot_cgges)(const char* jobvsl, const char* jobvsr, const char* sort, void* selctg, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, blas_int* sdim, void* alpha, void* beta, void* vsl, blas_int* ldvsl, void* vsr, blas_int* ldvsr, void* work, blas_int* lwork,  float* rwork,  float* bwork, blas_int* info);
  void coot_fortran(coot_zgges)(const char* jobvsl, const char* jobvsr, const char* sort, void* selctg, blas_int* n, void* a, blas_int* lda, void* b, blas_int* ldb, blas_int* sdim, void* alpha, void* beta, void* vsl, blas_int* ldvsl, void* vsr, blas_int* ldvsr, void* work, blas_int* lwork, double* rwork, double* bwork, blas_int* info);

  // 1-norm
  float  coot_fortran(coot_slange)(const char* norm, blas_int* m, blas_int* n,  float* a, blas_int* lda,  float* work);
  double coot_fortran(coot_dlange)(const char* norm, blas_int* m, blas_int* n, double* a, blas_int* lda, double* work);
  float  coot_fortran(coot_clange)(const char* norm, blas_int* m, blas_int* n,   void* a, blas_int* lda,  float* work);
  double coot_fortran(coot_zlange)(const char* norm, blas_int* m, blas_int* n,   void* a, blas_int* lda, double* work);

  // symmetric 1-norm
  float  coot_fortran(coot_slansy)(const char* norm, const char* uplo, const blas_int* N, float*  A, const blas_int* lda, float*  work);
  double coot_fortran(coot_dlansy)(const char* norm, const char* uplo, const blas_int* N, double* A, const blas_int* lda, double* work);
  float  coot_fortran(coot_clansy)(const char* norm, const char* uplo, const blas_int* N, void*   A, const blas_int* lda, float*  work);
  double coot_fortran(coot_zlansy)(const char* norm, const char* uplo, const blas_int* N, void*   A, const blas_int* lda, double* work);

  // reciprocal of condition number (real)
  void coot_fortran(coot_sgecon)(const char* norm, blas_int* n,  float* a, blas_int* lda,  float* anorm,  float* rcond,  float* work, blas_int* iwork, blas_int* info);
  void coot_fortran(coot_dgecon)(const char* norm, blas_int* n, double* a, blas_int* lda, double* anorm, double* rcond, double* work, blas_int* iwork, blas_int* info);

  // reciprocal of condition number (complex)
  void coot_fortran(coot_cgecon)(const char* norm, blas_int* n, void* a, blas_int* lda,  float* anorm,  float* rcond, void* work,  float* rwork, blas_int* info);
  void coot_fortran(coot_zgecon)(const char* norm, blas_int* n, void* a, blas_int* lda, double* anorm, double* rcond, void* work, double* rwork, blas_int* info);

  // obtain parameters according to the local configuration of lapack
  blas_int coot_fortran(coot_ilaenv)(blas_int* ispec, const char* name, const char* opts, blas_int* n1, blas_int* n2, blas_int* n3, blas_int* n4);

  // solve linear equations using LDL decomposition
  void coot_fortran(coot_ssytrs)(const char* uplo, blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, blas_int* ipiv, float*  b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_dsytrs)(const char* uplo, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_csytrs)(const char* uplo, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_zsytrs)(const char* uplo, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);

  // solve linear equations using LU decomposition
  void coot_fortran(coot_sgetrs)(const char* trans, blas_int* n, blas_int* nrhs, float*  a, blas_int* lda, blas_int* ipiv, float*  b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_dgetrs)(const char* trans, blas_int* n, blas_int* nrhs, double* a, blas_int* lda, blas_int* ipiv, double* b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_cgetrs)(const char* trans, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);
  void coot_fortran(coot_zgetrs)(const char* trans, blas_int* n, blas_int* nrhs, void*   a, blas_int* lda, blas_int* ipiv, void*   b, blas_int* ldb, blas_int* info);

  // calculate eigenvalues of an upper Hessenberg matrix
  void coot_fortran(coot_slahqr)(blas_int* wantt, blas_int* wantz, blas_int* n, blas_int* ilo, blas_int* ihi, float*  h, blas_int* ldh, float*  wr, float*  wi, blas_int* iloz, blas_int* ihiz, float*  z, blas_int* ldz, blas_int* info);
  void coot_fortran(coot_dlahqr)(blas_int* wantt, blas_int* wantz, blas_int* n, blas_int* ilo, blas_int* ihi, double* h, blas_int* ldh, double* wr, double* wi, blas_int* iloz, blas_int* ihiz, double* z, blas_int* ldz, blas_int* info);

  // calculate eigenvalues of a symmetric tridiagonal matrix
  void coot_fortran(coot_sstedc)(const char* compz, blas_int* n, float*  d, float*  e, float*  z, blas_int* ldz, float*  work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info);
  void coot_fortran(coot_dstedc)(const char* compz, blas_int* n, double* d, double* e, double* z, blas_int* ldz, double* work, blas_int* lwork, blas_int* iwork, blas_int* liwork, blas_int* info);

  // calculate eigenvectors of a Schur form matrix
  void coot_fortran(coot_strevc)(const char* side, const char* howmny, blas_int* select, blas_int* n, float*  t, blas_int* ldt, float*  vl, blas_int* ldvl, float*  vr, blas_int* ldvr, blas_int* mm, blas_int* m, float*  work, blas_int* info);
  void coot_fortran(coot_dtrevc)(const char* side, const char* howmny, blas_int* select, blas_int* n, double* t, blas_int* ldt, double* vl, blas_int* ldvl, double* vr, blas_int* ldvr, blas_int* mm, blas_int* m, double* work, blas_int* info);

  // generate a vector of random numbers
  void coot_fortran(coot_slarnv)(blas_int* idist, blas_int* iseed, blas_int* n, float*  x);
  void coot_fortran(coot_dlarnv)(blas_int* idist, blas_int* iseed, blas_int* n, double* x);

  // triangular factor of block reflector
  void coot_fortran(coot_slarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, float*  v, blas_int* ldv, float*  tau, float*  t, blas_int* ldt);
  void coot_fortran(coot_dlarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, double* v, blas_int* ldv, double* tau, double* t, blas_int* ldt);
  void coot_fortran(coot_clarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, void*   v, blas_int* ldv, void*   tau, void*   t, blas_int* ldt);
  void coot_fortran(coot_zlarft)(const char* direct, const char* storev, blas_int* n, blas_int* k, void*   v, blas_int* ldv, void*   tau, void*   t, blas_int* ldt);

  // generate an elementary reflector
  void coot_fortran(coot_slarfg)(const blas_int* n, float*  alpha, float*  x, const blas_int* incx, float*  tau);
  void coot_fortran(coot_dlarfg)(const blas_int* n, double* alpha, double* x, const blas_int* incx, double* tau);
  void coot_fortran(coot_clarfg)(const blas_int* n, void*   alpha, void*   x, const blas_int* incx, void*   tau);
  void coot_fortran(coot_zlarfg)(const blas_int* n, void*   alpha, void*   x, const blas_int* incx, void*   tau);

  // reduce a general matrix to bidiagonal form
  void coot_fortran(coot_sgebrd)(const blas_int* m, const blas_int* n, float*  a, const blas_int* lda, float*  d, float*  e, float*  tauq, float*  taup, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dgebrd)(const blas_int* m, const blas_int* n, double* a, const blas_int* lda, double* d, double* e, double* tauq, double* taup, double* work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_cgebrd)(const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, void*   d, void*   e, void*   tauq, void*   taup, void*   work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_zgebrd)(const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, void*   d, void*   e, void*   tauq, void*   taup, void*   work, const blas_int* lwork, blas_int* info);

  // generate Q or P**T determined by gebrd
  void coot_fortran(coot_sorgbr)(const char* vect, const blas_int* m, const blas_int* n, const blas_int* k, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dorgbr)(const char* vect, const blas_int* m, const blas_int* n, const blas_int* k, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info);

  // generate Q with orthonormal rows
  void coot_fortran(coot_sorglq)(const blas_int* m, const blas_int* n, const blas_int* k, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dorglq)(const blas_int* m, const blas_int* n, const blas_int* k, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info);

  // overwrite matrix with geqrf-generated orthogonal transformation
  void coot_fortran(coot_sormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dormqr)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info);

  // overwrite matrix with gelqf-generated orthogonal matrix
  void coot_fortran(coot_sormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dormlq)(const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info);

  // overwrite matrix with gebrd-generated orthogonal matrix products
  void coot_fortran(coot_sormbr)(const char* vect, const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const float*  A, const blas_int* lda, const float*  tau, float*  C, const blas_int* ldc, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dormbr)(const char* vect, const char* side, const char* trans, const blas_int* m, const blas_int* n, const blas_int* k, const double* A, const blas_int* lda, const double* tau, double* C, const blas_int* ldc, double* work, const blas_int* lwork, blas_int* info);

  // copy all or part of one 2d array to another
  void coot_fortran(coot_slacpy)(const char* uplo, const blas_int* m, const blas_int* n, const float*  A, const blas_int* lda, float*  B, const blas_int* ldb);
  void coot_fortran(coot_dlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const double* A, const blas_int* lda, double* B, const blas_int* ldb);
  void coot_fortran(coot_clacpy)(const char* uplo, const blas_int* m, const blas_int* n, const void*   A, const blas_int* lda, void*   B, const blas_int* ldb);
  void coot_fortran(coot_zlacpy)(const char* uplo, const blas_int* m, const blas_int* n, const void*   A, const blas_int* lda, void*   B, const blas_int* ldb);

  // initialize a matrix with different elements on and off the diagonal
  void coot_fortran(coot_slaset)(const char* uplo, const blas_int* m, const blas_int* n, const float*  alpha, const float*  beta, float*  A, const blas_int* lda);
  void coot_fortran(coot_dlaset)(const char* uplo, const blas_int* m, const blas_int* n, const double* alpha, const double* beta, double* A, const blas_int* lda);
  void coot_fortran(coot_claset)(const char* uplo, const blas_int* m, const blas_int* n, const void*   alpha, const void*   beta, void*   A, const blas_int* lda);
  void coot_fortran(coot_zlaset)(const char* uplo, const blas_int* m, const blas_int* n, const void*   alpha, const void*   beta, void*   A, const blas_int* lda);

  // apply block reflector to general rectangular matrix
  void coot_fortran(coot_slarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const float*  V, const blas_int* ldv, const float*  T, const blas_int* ldt, float*  C, const blas_int* ldc, float*  work, const blas_int* ldwork);
  void coot_fortran(coot_dlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const double* V, const blas_int* ldv, const double* T, const blas_int* ldt, double* C, const blas_int* ldc, double* work, const blas_int* ldwork);
  void coot_fortran(coot_clarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const void*   V, const blas_int* ldv, const void*   T, const blas_int* ldt, void*   C, const blas_int* ldc, void*   work, const blas_int* ldwork);
  void coot_fortran(coot_zlarfb)(const char* side, const char* trans, const char* direct, const char* storev, const blas_int* M, const blas_int* N, const blas_int* K, const void*   V, const blas_int* ldv, const void*   T, const blas_int* ldt, void*   C, const blas_int* ldc, void*   work, const blas_int* ldwork);

  // compute LQ factorization
  void coot_fortran(coot_sgelqf)(const blas_int* M, const blas_int* N, float*  A, const blas_int* lda, float*  tau, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dgelqf)(const blas_int* M, const blas_int* N, double* A, const blas_int* lda, double* tau, double* work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_cgelqf)(const blas_int* M, const blas_int* N, void*   A, const blas_int* lda, void*   tau, void*   work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_zgelqf)(const blas_int* M, const blas_int* N, void*   A, const blas_int* lda, void*   tau, void*   work, const blas_int* lwork, blas_int* info);

  // get machine parameters
  float  coot_fortran(coot_slamch)(const char* cmach);
  double coot_fortran(coot_dlamch)(const char* cmach);

  // scale matrix by a scalar
  void coot_fortran(coot_slascl)(const char* type, const blas_int* kl, const blas_int* ku, const float*  cfrom, const float*  cto, const blas_int* m, const blas_int* n, float*  a, const blas_int* lda, blas_int* info);
  void coot_fortran(coot_dlascl)(const char* type, const blas_int* kl, const blas_int* ku, const double* cfrom, const double* cto, const blas_int* m, const blas_int* n, double* a, const blas_int* lda, blas_int* info);
  void coot_fortran(coot_clascl)(const char* type, const blas_int* kl, const blas_int* ku, const void*   cfrom, const void*   cto, const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, blas_int* info);
  void coot_fortran(coot_zlascl)(const char* type, const blas_int* kl, const blas_int* ku, const void*   cfrom, const void*   cto, const blas_int* m, const blas_int* n, void*   a, const blas_int* lda, blas_int* info);

  // compute singular values of bidiagonal matrix
  void coot_fortran(coot_sbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, float*  vt, const blas_int* ldvt, float*  u, const blas_int* ldu, float*  c, const blas_int* ldc, float*  work, blas_int* info);
  void coot_fortran(coot_dbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, double* vt, const blas_int* ldvt, double* u, const blas_int* ldu, double* c, const blas_int* ldc, double* work, blas_int* info);
  void coot_fortran(coot_cbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, float*  d, float*  e, void*   vt, const blas_int* ldvt, void*   u, const blas_int* ldu, void*   c, const blas_int* ldc, float* work, blas_int* info);
  void coot_fortran(coot_zbdsqr)(const char* uplo, const blas_int* n, const blas_int* ncvt, const blas_int* nru, const blas_int* ncc, double* d, double* e, void*   vt, const blas_int* ldvt, void*   u, const blas_int* ldu, void*   c, const blas_int* ldc, double*  work, blas_int* info);

  // merges two sets of eigenvalues together into a single sorted set
  void coot_fortran(coot_slaed2)(blas_int* k, const blas_int* n, const blas_int* n1, float*  D, float*  Q, const blas_int* ldq, blas_int* indxq, float*  rho, const float*  Z, float*  dlamda, float*  W, float*  Q2, blas_int* indx, blas_int* indxc, blas_int* indxp, blas_int* coltyp, blas_int* info);
  void coot_fortran(coot_dlaed2)(blas_int* k, const blas_int* n, const blas_int* n1, double* D, double* Q, const blas_int* ldq, blas_int* indxq, double* rho, const double* Z, double* dlamda, double* W, double* Q2, blas_int* indx, blas_int* indxc, blas_int* indxp, blas_int* coltyp, blas_int* info);

  // compute all eigenvalues (and optionally eigenvectors) of symmetric tridiagonal matrix
  void coot_fortran(coot_ssteqr)(const char* compz, const blas_int* n, float*  D, float*  E, float*  Z, const blas_int* ldz, float*  work, blas_int* info);
  void coot_fortran(coot_dsteqr)(const char* compz, const blas_int* n, double* D, double* E, double* Z, const blas_int* ldz, double* work, blas_int* info);
  void coot_fortran(coot_csteqr)(const char* compz, const blas_int* n, void*   D, void*   E, void*   Z, const blas_int* ldz, void*   work, blas_int* info);
  void coot_fortran(coot_zsteqr)(const char* compz, const blas_int* n, void*   D, void*   E, void*   Z, const blas_int* ldz, void*   work, blas_int* info);

  // compute 1-norm/Frobenius norm/inf norm of real symmetric tridiagonal matrix
  float  coot_fortran(coot_slanst)(const char* norm, const blas_int* n, const float*  D, const float*  E);
  double coot_fortran(coot_dlanst)(const char* norm, const blas_int* n, const double* D, const double* E);

  // reduce real symmetric matrix to tridiagonal form
  void coot_fortran(coot_ssytrd)(const char* uplo, const blas_int* n, float*  A, const blas_int* lda, float*  D, float*  E, float*  tau, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dsytrd)(const char* uplo, const blas_int* n, double* A, const blas_int* lda, double* D, double* E, double* tau, double* work, const blas_int* lwork, blas_int* info);

  // force A and B to be stored prior to doing the addition of A and B
  float  coot_fortran(coot_slamc3)(const float*  A, const float*  B);
  double coot_fortran(coot_dlamc3)(const double* A, const double* B);

  // compute the i'th updated eigenvalue of a symmetric rank-one modification to the diagonal matrix in d
  void coot_fortran(coot_slaed4)(const blas_int* n, const blas_int* i, const float*  D, const float*  Z, float*  delta, const float*  rho, float*  dlam, const blas_int* info);
  void coot_fortran(coot_dlaed4)(const blas_int* n, const blas_int* i, const double* D, const double* Z, double* delta, const double* rho, double* dlam, const blas_int* info);

  // create a permutation list to merge the element of A into a single set
  void coot_fortran(coot_slamrg)(const blas_int* n1, const blas_int* n2, const float*  A, const blas_int* dtrd1, const blas_int* dtrd2, blas_int* index);
  void coot_fortran(coot_dlamrg)(const blas_int* n1, const blas_int* n2, const double* A, const blas_int* dtrd1, const blas_int* dtrd2, blas_int* index);

  // generate real orthogonal matrix as the product of dsytrd-generated elementary reflectors
  void coot_fortran(coot_sorgtr)(const char* uplo, const blas_int* n, float*  A, const blas_int* lda, const float*  tau, float*  work, const blas_int* lwork, blas_int* info);
  void coot_fortran(coot_dorgtr)(const char* uplo, const blas_int* n, double* A, const blas_int* lda, const double* tau, double* work, const blas_int* lwork, blas_int* info);
  }
