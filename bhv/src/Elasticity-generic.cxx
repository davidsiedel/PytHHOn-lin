/*!
* \file   Elasticity-generic.cxx
* \brief  This file implements the umat interface for the Elasticity behaviour law
* \author Helfer Thomas
* \date   23 / 11 / 06
*/

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif /* NOMINMAX */
#include <windows.h>
#ifdef small
#undef small
#endif /* small */
#endif /* _WIN32 */

#ifndef MFRONT_SHAREDOBJ
#define MFRONT_SHAREDOBJ TFEL_VISIBILITY_EXPORT
#endif /* MFRONT_SHAREDOBJ */

#include<iostream>
#include<cstdlib>
#include"TFEL/Material/OutOfBoundsPolicy.hxx"
#include"TFEL/Math/Tensor/TensorView.hxx"
#include"TFEL/Math/Stensor/StensorView.hxx"
#include"TFEL/Math/Stensor/StensorView.hxx"
#include"TFEL/Math/Tensor/TensorView.hxx"
#include"TFEL/Math/t2tot2.hxx"
#include"TFEL/Math/T2toT2/T2toT2View.hxx"
#include"TFEL/Math/t2tost2.hxx"
#include"TFEL/Math/T2toST2/T2toST2View.hxx"
#include"TFEL/Material/Elasticity.hxx"
#include"MFront/GenericBehaviour/Integrate.hxx"

#include"MFront/GenericBehaviour/Elasticity-generic.hxx"

static tfel::material::OutOfBoundsPolicy&
Elasticity_getOutOfBoundsPolicy(){
using namespace tfel::material;
static OutOfBoundsPolicy policy = None;
return policy;
}

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

MFRONT_SHAREDOBJ const char* 
Elasticity_build_id = "";

MFRONT_SHAREDOBJ const char* 
Elasticity_mfront_ept = "Elasticity";

MFRONT_SHAREDOBJ const char* 
Elasticity_tfel_version = "3.4.0-dev";

MFRONT_SHAREDOBJ unsigned short Elasticity_mfront_mkt = 1u;

MFRONT_SHAREDOBJ const char *
Elasticity_mfront_interface = "Generic";

MFRONT_SHAREDOBJ const char *
Elasticity_src = "Elasticity.mfront";

MFRONT_SHAREDOBJ unsigned short Elasticity_nModellingHypotheses = 5u;

MFRONT_SHAREDOBJ const char * 
Elasticity_ModellingHypotheses[5u] = {"AxisymmetricalGeneralisedPlaneStrain",
"Axisymmetrical",
"PlaneStrain",
"GeneralisedPlaneStrain",
"Tridimensional"};

MFRONT_SHAREDOBJ unsigned short Elasticity_nMainVariables = 1;
MFRONT_SHAREDOBJ unsigned short Elasticity_nGradients = 1;

MFRONT_SHAREDOBJ int Elasticity_GradientsTypes[1] = {1};
MFRONT_SHAREDOBJ const char * Elasticity_Gradients[1] = {"Strain"};
MFRONT_SHAREDOBJ unsigned short Elasticity_nThermodynamicForces = 1;

MFRONT_SHAREDOBJ int Elasticity_ThermodynamicForcesTypes[1] = {1};
MFRONT_SHAREDOBJ const char * Elasticity_ThermodynamicForces[1] = {"Stress"};
MFRONT_SHAREDOBJ unsigned short Elasticity_nTangentOperatorBlocks = 2;

MFRONT_SHAREDOBJ const char * Elasticity_TangentOperatorBlocks[2] = {"Stress",
"Strain"};
MFRONT_SHAREDOBJ unsigned short Elasticity_BehaviourType = 1u;

MFRONT_SHAREDOBJ unsigned short Elasticity_BehaviourKinematic = 1u;

MFRONT_SHAREDOBJ unsigned short Elasticity_SymmetryType = 0u;

MFRONT_SHAREDOBJ unsigned short Elasticity_ElasticSymmetryType = 0u;

MFRONT_SHAREDOBJ unsigned short Elasticity_UsableInPurelyImplicitResolution = 1;

MFRONT_SHAREDOBJ unsigned short Elasticity_nMaterialProperties = 2u;

MFRONT_SHAREDOBJ const char *Elasticity_MaterialProperties[2u] = {"YoungModulus",
"PoissonRatio"};

MFRONT_SHAREDOBJ unsigned short Elasticity_nInternalStateVariables = 0;
MFRONT_SHAREDOBJ const char * const * Elasticity_InternalStateVariables = nullptr;

MFRONT_SHAREDOBJ const int * Elasticity_InternalStateVariablesTypes = nullptr;

MFRONT_SHAREDOBJ unsigned short Elasticity_nExternalStateVariables = 0;
MFRONT_SHAREDOBJ const char * const * Elasticity_ExternalStateVariables = nullptr;

MFRONT_SHAREDOBJ unsigned short Elasticity_nParameters = 2;
MFRONT_SHAREDOBJ const char * Elasticity_Parameters[2] = {"minimal_time_step_scaling_factor",
"maximal_time_step_scaling_factor"};
MFRONT_SHAREDOBJ int Elasticity_ParametersTypes [] = {0,0};

MFRONT_SHAREDOBJ double Elasticity_minimal_time_step_scaling_factor_ParameterDefaultValue = 0.1;

MFRONT_SHAREDOBJ double Elasticity_maximal_time_step_scaling_factor_ParameterDefaultValue = 1.7976931348623e+308;

MFRONT_SHAREDOBJ unsigned short Elasticity_requiresStiffnessTensor = 0;
MFRONT_SHAREDOBJ unsigned short Elasticity_requiresThermalExpansionCoefficientTensor = 0;

MFRONT_SHAREDOBJ unsigned short Elasticity_ComputesInternalEnergy = 0;

MFRONT_SHAREDOBJ unsigned short Elasticity_ComputesDissipatedEnergy = 0;

MFRONT_SHAREDOBJ void
Elasticity_setOutOfBoundsPolicy(const int p){
if(p==0){
Elasticity_getOutOfBoundsPolicy() = tfel::material::None;
} else if(p==1){
Elasticity_getOutOfBoundsPolicy() = tfel::material::Warning;
} else if(p==2){
Elasticity_getOutOfBoundsPolicy() = tfel::material::Strict;
} else {
std::cerr << "Elasticity_setOutOfBoundsPolicy: invalid argument\n";
}
}

MFRONT_SHAREDOBJ int
Elasticity_setParameter(const char *const key,const double value){
using tfel::material::ElasticityParametersInitializer;
auto& i = ElasticityParametersInitializer::get();
try{
i.set(key,value);
} catch(std::runtime_error& e){
std::cerr << e.what() << std::endl;
return 0;
}
return 1;
}

MFRONT_SHAREDOBJ int Elasticity_AxisymmetricalGeneralisedPlaneStrain(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr const auto h = ModellingHypothesis::AXISYMMETRICALGENERALISEDPLANESTRAIN;
using Behaviour = Elasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, Elasticity_getOutOfBoundsPolicy());
return r;
} // end of Elasticity_AxisymmetricalGeneralisedPlaneStrain

MFRONT_SHAREDOBJ int Elasticity_Axisymmetrical(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr const auto h = ModellingHypothesis::AXISYMMETRICAL;
using Behaviour = Elasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, Elasticity_getOutOfBoundsPolicy());
return r;
} // end of Elasticity_Axisymmetrical

MFRONT_SHAREDOBJ int Elasticity_PlaneStrain(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr const auto h = ModellingHypothesis::PLANESTRAIN;
using Behaviour = Elasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, Elasticity_getOutOfBoundsPolicy());
return r;
} // end of Elasticity_PlaneStrain

MFRONT_SHAREDOBJ int Elasticity_GeneralisedPlaneStrain(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr const auto h = ModellingHypothesis::GENERALISEDPLANESTRAIN;
using Behaviour = Elasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, Elasticity_getOutOfBoundsPolicy());
return r;
} // end of Elasticity_GeneralisedPlaneStrain

MFRONT_SHAREDOBJ int Elasticity_Tridimensional(mfront_gb_BehaviourData* const d){
using namespace tfel::material;
using real = mfront::gb::real;
constexpr const auto h = ModellingHypothesis::TRIDIMENSIONAL;
using Behaviour = Elasticity<h,real,false>;
const auto r = mfront::gb::integrate<Behaviour>(*d,Behaviour::STANDARDTANGENTOPERATOR, Elasticity_getOutOfBoundsPolicy());
return r;
} // end of Elasticity_Tridimensional

#ifdef __cplusplus
}
#endif /* __cplusplus */

