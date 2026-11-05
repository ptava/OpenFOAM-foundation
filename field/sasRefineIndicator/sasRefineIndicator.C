/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2014-2016 OpenFOAM Foundation
    Copyright (C) 2020 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/
#include "sasRefineIndicator.H"
#include "fvcGrad.H"
#include "addToRunTimeSelectionTable.H"
#include "IOobjectList.H"

#include <cmath>

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(sasRefineIndicator, 0);
    addToRunTimeSelectionTable(functionObject, sasRefineIndicator, dictionary);
namespace
{

// --- Default values
const word defaultResultName = "sasRefineIndicator";
const scalar sigmaDefault = 0.05;
const scalar weight1Default = 0.0;
const scalar weight2Default = 1.0;
const scalar alphaDefault = 0.6;


// --- Mathematical helpers
inline scalar oneMinusExpNeg(const scalar x)
{
    return (mag(x) < 1e-6) ? x - 0.5*sqr(x) : -expm1(-x);
}

// For b*(1-c*b^p), the peak is [c*(p+1)]^(-1/p).
scalar logEndpointRatioMagnitude
(
    const scalar logEnd,
    const scalar p,
    const bool negativeEnd
)
{
    const scalar logDamping = negativeEnd ?
        std::log1p(exp(logEnd)) : log(-expm1(logEnd));

    return
        logEnd
      + (logDamping + std::log1p(p))/p
      + std::log1p(1/p);
}


// Return 1-c, preserving endpoint precision when c is close to 1.
scalar dampingEndpoint(const scalar alpha)
{
    if (alpha == 1)
    {
        return 0;
    }

    const scalar p = 2/alpha;
    const bool negativeEnd = alpha > 1;
    const scalar logTarget = negativeEnd ?
        log(alpha - 1) : std::log1p(-alpha);
    scalar hi = negativeEnd ? 0 : -std::log1p(1/p);

    while
    (
        negativeEnd
     && logEndpointRatioMagnitude(hi, p, negativeEnd) < logTarget
    )
    {
        hi += 1;
    }

    scalar lo = hi - 1;
    while (logEndpointRatioMagnitude(lo, p, negativeEnd) > logTarget)
    {
        lo -= max(scalar(1), mag(lo));
    }

    for (label iter = 0; iter < 100; ++iter)
    {
        const scalar mid = 0.5*(lo + hi);
        if (logEndpointRatioMagnitude(mid, p, negativeEnd) < logTarget)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }

    const scalar logEnd = 0.5*(lo + hi);
    const scalar error =
        logEndpointRatioMagnitude(logEnd, p, negativeEnd) - logTarget;

    if (!std::isfinite(error) || mag(error) > sqrt(SMALL))
    {
        FatalErrorInFunction
            << "Cannot resolve safeScaler damping for alpha = " << alpha
            << exit(FatalError);
    }

    return negativeEnd ? -exp(logEnd) : exp(logEnd);
}

} // End anonymous namespace

const NamedEnum<sasRefineIndicator::focusRegion, 2>
sasRefineIndicator::focusRegionNames_
{
    "core",
    "periphery"
};

const NamedEnum<sasRefineIndicator::transferFunction, 5>
sasRefineIndicator::transferFunctionNames_
{
    "basic",
    "constant",
    "oddScaler",
    "safeScaler",
    "gaussSink"
};

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

sasRefineIndicator::sasRefineIndicator
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    zone_(dict.lookup("cellZone")),
    focusRegion_(focusRegion::core),
    transferFunction_(transferFunction::constant),
    sigma_(sigmaDefault),
    weight1_(weight1Default),
    weight2_(weight2Default),
    alpha_(alphaDefault),
    controlRunTime_(false),
    resultName_(defaultResultName)
{
    read(dict);

    auto* fldPtr = new volScalarField
    (
        IOobject
        (
            resultName_,
            mesh_.time().name(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh_,
        dimensionedScalar(dimless, -GREAT)
    );
    mesh_.objectRegistry::store(fldPtr);
}

// * * * * * * * * * * * * * * * *  Helpers  * * * * * * * * * * * * * * * * //

tmp<volScalarField::Internal> sasRefineIndicator::makeTmpInternal
(
    const word& name,
    const dimensionedScalar& init
) const
{
    return tmp<volScalarField::Internal>
    (
        new volScalarField::Internal
        (
            IOobject
            (
                name,
                mesh_.time().name(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            mesh_,
            init
        )
    );
}

tmp<volScalarField::Internal> sasRefineIndicator::markCoreBasic
(
    const cellZone& cells,
    const volScalarField::Internal& c1,
    const volScalarField::Internal& c2
) const
{
    tmp<volScalarField::Internal> tG =
        makeTmpInternal
        (
            "tmpG",
            dimensionedScalar(dimless, -GREAT)
        );
    volScalarField::Internal& G = tG.ref();

    for(const label i : cells)
    {
        // Calculate the difference between c2 and c1 for each cell
        // and apply constant value to all positive values.
        scalar d = c2[i] - c1[i];
        G[i] = d > 0.0 ? d : -GREAT;
    }

    return tG;
}

tmp<volScalarField::Internal> sasRefineIndicator::markCoreConstant
(
    const cellZone& cells,
    const volScalarField::Internal& c1,
    const volScalarField::Internal& c2,
    const scalar weight
) const
{
    tmp<volScalarField::Internal> tG =
        makeTmpInternal
        (
            "tmpG",
            dimensionedScalar(dimless, -GREAT)
        );
    volScalarField::Internal& G = tG.ref();

    for(const label i : cells)
    {
        // Calculate the difference between c2 and c1 for each cell
        // and apply constant value to all positive values.
        scalar d = c2[i] - c1[i];
        G[i] = d > 0.0 ? weight : -GREAT;
    }

    return tG;
}

tmp<volScalarField::Internal> sasRefineIndicator::markCoreOddScaler
(
    const cellZone& cells,
    const volScalarField::Internal& c1,
    const volScalarField::Internal& c2,
    const scalar weight,
    const scalar sigma
) const
{
    const scalar invTwoSigma = 0.5/sqr(sigma);

    tmp<volScalarField::Internal> tG =
        makeTmpInternal
        (
            "tmpG",
            dimensionedScalar(dimless, -GREAT)
        );
    volScalarField::Internal& G = tG.ref();
    const tmp<volScalarField::Internal> td(c2 - c1);
    const volScalarField::Internal& d = td.ref();

    scalar dMaxLocal = -GREAT;
    for (const label i : cells)
    {
        dMaxLocal = max(dMaxLocal, d[i]);
    }
    const scalar dMax = returnReduce(dMaxLocal, maxOp());

    DebugInfo
        << "Calculated dMax = " << dMax << " for zone " << zone_ << nl;

    if (dMax <= SMALL)
    {
        return tG;
    }

    const scalar shift = weight * dMax;
    for (const label celli : cells)
    {
        // Calculate the difference between c2 and c1 for each cell
        // and apply an odd, monotonic, sign-preserving function
        const scalar u = d[celli] + shift;
        const scalar x = invTwoSigma * sqr(u);
        G[celli] = u * oneMinusExpNeg(x);
    }
    // Normalise by maximum value of G for d > -shift
    // const scalar maxG = gMax(G.primitiveField());
    // for (const label i : cells)
    // {
    //     const scalar u = d[i] + shift;
    //     const scalar Gi = G[i];
    //     G[i] = (u > 0.0) ? Gi/maxG : Gi;
    // }
    // Normalise by maximum value of G
    const scalar maxG = gMax(G.primitiveField());
    if (maxG > SMALL)
    {
        G /= maxG;
    }

    return tG;
}

tmp<volScalarField::Internal>
sasRefineIndicator::markCoreSafeScaler
(
    const cellZone& cells,
    const volScalarField::Internal& c1,
    const volScalarField::Internal& c2,
    const scalar weight,
    const scalar sigma,
    const scalar alpha
) const
{
    if (!std::isfinite(alpha) || alpha < 0)
    {
        FatalErrorInFunction
            << "'alpha' must be finite and non-negative (got " << alpha << ")."
            << exit(FatalError);
    }

    if
    (
        !std::isfinite(sigma) || sigma <= SMALL
     || !std::isfinite(weight) || weight < 0
    )
    {
        FatalErrorInFunction
            << "safeScaler requires finite sigma > 0 and weight1 >= 0."
            << exit(FatalError);
    }

    // Values below scalar precision have no distinguishable endpoint damping.
    if (1 - alpha == 1)
    {
        return markCoreOddScaler(cells, c1, c2, weight, sigma);
    }

    tmp<volScalarField::Internal> tG =
        makeTmpInternal
        (
            "tmpG",
            dimensionedScalar(dimless, -GREAT)
        );
    volScalarField::Internal& G = tG.ref();
    const tmp<volScalarField::Internal> td(c2 - c1);
    const volScalarField::Internal& d = td.ref();

    scalar dMaxLocal = -GREAT;
    for (const label celli : cells)
    {
        dMaxLocal = max(dMaxLocal, d[celli]);
    }
    const scalar dMax = returnReduce(dMaxLocal, maxOp<scalar>());

    DebugInfo
        << "Calculated dMax = " << dMax << " for zone " << zone_ << nl;

    if (dMax <= SMALL)
    {
        return tG;
    }

    const scalar shift = weight*dMax;
    const scalar uMax = dMax + shift;
    const scalar invTwoSigma2 = 0.5/sqr(sigma);
    const scalar endFactor = oneMinusExpNeg(sqr(uMax)*invTwoSigma2);

    if (!std::isfinite(uMax) || !(endFactor > 0))
    {
        FatalErrorInFunction
            << "Cannot scale the oddScaler endpoint for zone " << zone_
            << ". Check sigma, weight1 and the input field."
            << exit(FatalError);
    }

    const scalar endpoint = dampingEndpoint(alpha);
    scalar maxGLocal = 0;

    for (const label celli : cells)
    {
        const scalar u = d[celli] + shift;
        const scalar b = d[celli] == dMax ? 1 :
            (u/uMax)*oneMinusExpNeg(sqr(u)*invTwoSigma2)/endFactor;

        G[celli] = b;
        if (b > 0)
        {
            const scalar logPower = b < 1 ? 2*log(b)/alpha : 0;
            const scalar power = exp(logPower);
            // endpoint = 1-c. This form preserves precision near alpha=1
            // when evaluating b*(1-c*b^p), with p=2/alpha.
            G[celli] *= oneMinusExpNeg(-logPower) + endpoint*power;
        }
        maxGLocal = max(maxGLocal, G[celli]);
    }

    const scalar maxG = returnReduce(maxGLocal, maxOp<scalar>());

    if (!std::isfinite(maxG) || !(maxG > VSMALL))
    {
        FatalErrorInFunction
            << "safeScaler has no positive maximum in zone " << zone_
            << " after damping; the field cannot be normalised."
            << exit(FatalError);
    }

    for (const label celli : cells)
    {
        G[celli] /= maxG;
    }

    return tG;
}


tmp<volScalarField::Internal> sasRefineIndicator::markCoreGaussSink
(
    const cellZone& cells,
    const volScalarField::Internal& Lvk,
    const volScalarField::Internal& c2,
    const scalar weight,
    const scalar sigma
) const
{
    const scalar invTwoSigma = 0.5/(sqr(sigma));

    tmp<volScalarField::Internal> tG =
        makeTmpInternal
        (
            "tmpG",
            dimensionedScalar(dimless, -GREAT)
        );
    tmp<volScalarField::Internal> tnLvk = Lvk / 
        ( c2 + dimensionedScalar(dimLength, VSMALL));

    auto& G = tG.ref();
    auto& nLvk = tnLvk.ref();

    for(const label i : cells)
    {
        // Calculate normalised von Karman length scale for each cell
        // and apply the Gaussian function to get the indicator value.
        const scalar& nLvki = nLvk[i];
        G[i] = exp(-invTwoSigma * sqr(nLvki - 1))
             - weight * sqr(nLvki - 1);
    }

    return tG;
}

tmp<volScalarField::Internal> sasRefineIndicator::markPeripheryGaussSink
(
    const cellZone& cells,
    const volScalarField::Internal& Lvk,
    const scalar LvkRef,
    const scalar weight,
    const scalar sigma
) const
{
    const scalar invTwoSigma = 0.5/(sqr(sigma));

    tmp<volScalarField::Internal> tG =
        makeTmpInternal
        (
            "tmpG",
            dimensionedScalar(dimless, -GREAT)
        );
    tmp<volScalarField::Internal> tnLvk = Lvk / LvkRef;

    auto& G = tG.ref();
    auto& nLvk = tnLvk.ref();

    for(const label i : cells)
    {
        // Calculate normalised von Karman length scale for each cell
        // and apply the Gaussian function to get the indicator value.
        const scalar& nLvki = nLvk[i];
        G[i] = exp(-invTwoSigma * sqr(nLvki - 1))
             - weight * sqr(nLvki - 1);
    }

    return tG;
}

// * * * * * * * * * * * * * *  Calculation  * * * * * * * * * * * * * * * * //

void sasRefineIndicator::calcIndicator()
{
    const cellZone& cells = mesh().cellZones()[zone_];

    if (debug)
    {
        Pout
            << "Cells in zone " << zone_ << ": "
            << cells.size() << endl;
    }

    auto& fld = mesh_.lookupObjectRef<volScalarField>(resultName_);

    auto& fldI = fld.internalFieldRef();

    switch (focusRegion_)
    {
        case focusRegion::core:
        {
            const volScalarField& C1  = lookupObject<volScalarField>("C1");
            const volScalarField& C2  = lookupObject<volScalarField>("C2");
            const auto& C1I  = C1.internalField();
            const auto& C2I  = C2.internalField();
            switch (transferFunction_)
            {
                case transferFunction::basic:
                {
                    fldI = markCoreBasic(cells, C1I, C2I);
                    break;
                }
                case transferFunction::constant:
                {
                    fldI = markCoreConstant(cells, C1I, C2I, weight2_);
                    break;
                }
                case transferFunction::oddScaler:
                {
                    fldI = markCoreOddScaler(cells, C1I, C2I, weight1_, sigma_);
                    break;
                }
                case transferFunction::safeScaler:
                {
                    // TO DO: read dynamically from dict
                    // if (controlRunTime_)
                    // {
                    //     alpha_ = readScalar(dict().lookup("alpha"));
                    // }
                    fldI = markCoreSafeScaler(cells, C1I, C2I, weight1_, sigma_, alpha_);
                    break;
                }
                case transferFunction::gaussSink:
                {
                    const volScalarField& Lvk = lookupObject<volScalarField>("Lvk");
                    const auto& LvkI = Lvk.internalField();
                    fldI = markCoreGaussSink
                    (
                        cells,
                        LvkI,
                        C2I,
                        weight2_,
                        sigma_
                    );
                    break;
                }
            }
            break;
        }
        case focusRegion::periphery:
        {
            const volScalarField& Lvk = lookupObject<volScalarField>("Lvk");
            const auto& LvkI = Lvk.internalField();
            fldI = markPeripheryGaussSink
            (
                cells,
                LvkI,
                LvkRef_,
                weight2_,
                sigma_
            );
            break;
        }
    }

    // Debug info - range of indicator field
    DebugInfo
        << name() << " (" << resultName_ << "): range = ["
        << gMin(fld.internalField()) << ", "
        << gMax(fld.internalField()) << "]" << nl;

    // Debug info - number of cells with indicator >= 0
    if (debug)
    {
        label nPos = 0;
        forAll(fldI, i)
        {
            if (fldI[i] >= 0)
            {
                nPos++;
            }
        }
        DebugInfo
            << name() << " (" << resultName_ << "): found "
            << returnReduce(nPos, sumOp()) << "/"
            << returnReduce(cells.size(), sumOp())
            << " cells with indicator >= 0: " << endl;
    }
}

// * * * * * * * * * * * * * *  Read/Execute/Write  * * * * * * * * * * * * * //

bool sasRefineIndicator::read(const dictionary& dict)
{
    if (!fvMeshFunctionObject::read(dict))
    {
        return false;
    }

    requiredFields_.clear();
    resultName_ = dict.lookupOrDefault<word>("result", defaultResultName);
    sigma_ = dict.lookupOrDefault<scalar>("sigma", sigmaDefault);
    focusRegion_ = focusRegionNames_.read(dict.lookup("focusRegion"));

    switch (focusRegion_)
    {
        case focusRegion::core:
        {
            requiredFields_.append("C1");
            requiredFields_.append("C2");
            requiredFields_.append("Lvk");
            weight1_ = dict.lookupOrDefault<scalar>("weight1", weight1Default);
            weight2_ = dict.lookupOrDefault<scalar>("weight2", weight2Default);
            alpha_ = dict.lookupOrDefault<scalar>("alpha", alphaDefault);
            transferFunction_ = transferFunctionNames_.read(
                dict.lookup("transferFunction")
            );

            if
            (
                transferFunction_ == transferFunction::safeScaler
             && (!std::isfinite(alpha_) || alpha_ < 0)
            )
            {
                FatalIOErrorInFunction(dict)
                    << "'alpha' must be finite and non-negative (got "
                    << alpha_ << ")." << exit(FatalIOError);
            }

            break;
        }
        case focusRegion::periphery:
        {
            requiredFields_.append("Lvk");
            weight1_ = dict.lookupOrDefault<scalar>("weight1", weight1Default);
            weight2_ = dict.lookupOrDefault<scalar>("weight2", weight2Default);
            LvkRef_ = dict.lookup<scalar>("LvkRef");
            break;
        }
        default:
        {
            FatalIOErrorInFunction(dict)
                << "Unknown 'focusRegion' " << focusRegionNames_[focusRegion_]
                << exit(FatalIOError);
        }
    }

    if (sigma_ <= SMALL)
    {
        FatalIOErrorInFunction(dict)
            << "'sigma' must be > 0 (got " << sigma_ << ")." 
            << exit(FatalIOError);
    }

    if (weight1_ < 0.0 || weight2_ < 0.0)
    {
        FatalIOErrorInFunction(dict)
            << "'weight1' and 'weight2' must be non-negative (got "
            << weight1_ << " and " << weight2_ << ")."
            << exit(FatalIOError);
    }

    if (log)
    {
        Info<< type() << ' ' << name() << ':' << nl
            << "  focusRegion      : " << focusRegionNames_[focusRegion_] << nl
            << "  transferFunction : " << transferFunctionNames_[transferFunction_] << nl
            << "  sigma            : " << sigma_ << nl
            << "  weight1          : " << weight1_ << nl
            << "  weight2          : " << weight2_ << nl
            << "  alpha            : " << alpha_ << nl
            << "  result           : " << resultName_ << nl
            << "  nCells           : " << mesh().cellZones()[zone_].size() << nl
            << endl;
    }

    return true;
}

wordList sasRefineIndicator::fields() const
{
    return requiredFields_;
}

bool sasRefineIndicator::execute()
{
    for (const auto& fieldName : requiredFields_)
    {
        if (!foundObject<volScalarField>(fieldName))
        {
            FatalErrorInFunction
                << "Required field '" << fieldName << "' not found."
                << "Available volScalarField objects in object registry: "
                << obr_.lookupClass<volScalarField>().toc()
                << exit(FatalError);
        }
    }

    calcIndicator();
    return true;
}

bool sasRefineIndicator::write()
{
    Info
        << "FunctionObjects::" << name() 
        << " writing field: " << resultName_ << endl;

    const auto& fld = mesh_.lookupObject<volScalarField>(resultName_);
    fld.write();
    return true;
}

} // End namespace functionObjects
} // End namespace Foam
