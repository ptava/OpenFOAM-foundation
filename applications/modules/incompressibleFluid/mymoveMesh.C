/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2026 OpenFOAM Foundation
     \\/     M anipulation  |
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

#include "myincompressibleFluid.H"
#include "fvCorrectPhi.H"
#include "fvcMeshPhi.H"
#include "fviDiv.H"
#include "Tuple2.H"
#include "IOdictionary.H"
#include "clock.H"
#include "OSspecific.H"
#include <cmath>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{
    const int correctPhiDebug = debug::debugSwitch
    (
        "correctPhi",
        0
    );
}
}


// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //

void Foam::solvers::myincompressibleFluid::writeCorrectPhiContinuity
(
    const word& stage,
    const scalar threshold,
    const fileName& outputPath
) const
{
    const volInternalScalarField residual(fvi::div(phi_));
    const scalarField& V = mesh.V();

    bool invalid = false;
    label cellsAbove = 0;
    Tuple2<scalar, vector> maximum(-great, vector(Zero));

    forAll(residual, celli)
    {
        const scalar magResidual = mag(residual[celli]);
        invalid =
            invalid
         || !std::isfinite(magResidual)
         || !std::isfinite(V[celli])
         || V[celli] <= 0;

        if (magResidual > maximum.first())
        {
            maximum.first() = magResidual;
            maximum.second() = mesh.C()[celli];
        }
        if (magResidual > threshold)
        {
            ++cellsAbove;
        }
    }

    reduce(invalid, orOp());
    if (invalid)
    {
        FatalErrorInFunction
            << "Nonfinite residual or invalid cell volume"
            << exit(FatalError);
    }

    reduce
    (
        maximum,
        [](const Tuple2<scalar, vector>& a, const Tuple2<scalar, vector>& b)
        {
            return a.first() >= b.first() ? a : b;
        }
    );
    reduce(cellsAbove, sumOp());

    if (maximum.first() < 0)
    {
        FatalErrorInFunction
            << "Cannot compute continuity statistics on an empty mesh"
            << exit(FatalError);
    }

    const scalar meanAbs =
        weightedAverage(mag(residual), mesh.V()).value();
    if (!std::isfinite(meanAbs))
    {
        FatalErrorInFunction
            << "Nonfinite volume-average residual" << exit(FatalError);
    }

    Info<< "continuity stage=" << stage
        << " cellsAbove=" << cellsAbove
        << " maxAbs=" << maximum.first()
        << " maxLocation=" << maximum.second()
        << " volAverageAbs=" << meanAbs << endl;

    if (!outputPath.empty())
    {
        writeCorrectPhiResidual(residual, outputPath/stage);
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::myincompressibleFluid::motionCorrector()
{
    if (pimple.firstIter() || pimple.moveMeshOuterCorrectors())
    {
        if (mesh.changing())
        {
            if (correctPhi || mesh.poly().topoChanged())
            {
                // Calculate absolute flux
                // from the mapped surface velocity
                phi_ = mesh.Sf() & Uf();

                correctUphiBCs(U_, phi_, true);

                const bool diagnose =
                    correctPhiDebug
                 && mesh.poly().topoChanged();
                scalar threshold = 0;
                fileName outputPath;
                if (diagnose)
                {
                    if (correctPhiDebug < 1 || correctPhiDebug > 2)
                    {
                        FatalErrorInFunction
                            << "correctPhi debug level must be 0, 1 or 2"
                            << exit(FatalError);
                    }
                    if
                    (
                        mesh.poly().moving() || MRF.size()
                     || phi_.dimensions()
                        != dimensions::volume/dimensions::time
                    )
                    {
                        FatalErrorInFunction
                            << "Diagnostics require a stationary mesh, "
                            << "no MRF and volumetric phi"
                            << exit(FatalError);
                    }
                    const IOdictionary controls
                    (
                        IOobject
                        (
                            "continuityDiagnostics",
                            runTime.system(),
                            mesh,
                            IOobject::MUST_READ,
                            IOobject::NO_WRITE,
                            false
                        )
                    );
                    threshold = dimensionedScalar
                    (
                        "threshold",
                        dimless/dimensions::time,
                        controls
                    ).value();
                    if (!std::isfinite(threshold) || threshold < 0)
                    {
                        FatalIOErrorInFunction(controls)
                            << "threshold must be finite and non-negative"
                            << exit(FatalIOError);
                    }
                    if (correctPhiDebug == 2)
                    {
                        if (continuityOutput_.empty())
                        {
                            if (Pstream::master())
                            {
                                const fileName base = runTime.globalPath()
                                   /"postProcessing"/"continuity"/mesh.name();
                                const word segment =
                                    Foam::name(clock::getTime())
                                  + '-' + Foam::name(Foam::pid());
                                continuityOutput_ = base/segment;
                                label suffix = 0;
                                while (isDir(continuityOutput_))
                                {
                                    continuityOutput_ =
                                        base
                                       /(segment + '-' + Foam::name(++suffix));
                                }
                            }
                            Pstream::scatter(continuityOutput_);
                        }
                        outputPath =
                            continuityOutput_/Foam::name(++continuityCapture_);
                    }
                    writeCorrectPhiContinuity
                    (
                        "beforeCorrectPhi", threshold, outputPath
                    );
                }

                fv::correctPhi
                (
                    phi_,
                    U,
                    p,
                    autoPtr<volScalarField>(),
                    autoPtr<volInternalScalarField>(),
                    pressureReference,
                    pimple
                );

                if (diagnose)
                {
                    writeCorrectPhiContinuity
                    (
                        "afterCorrectPhi", threshold, outputPath
                    );
                }

                // Make the flux relative to the mesh motion
                MRF.makeRelative(phi_);
                fvc::makeRelative(phi_, U);
            }

            meshCourantNo();
        }
    }
}


// ************************************************************************* //
