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

Class
    Foam::solvers::myincompressibleFluid

Description
    Incompressible fluid solver with optional continuity statistics immediately
    before and after mesh-change flux correction.

    All other solver behaviour is inherited from incompressibleFluid.

SourceFiles
    myincompressibleFluid.C
    mymoveMesh.C

\*---------------------------------------------------------------------------*/

#ifndef myincompressibleFluid_H
#define myincompressibleFluid_H

#include "incompressibleFluid.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{

/*---------------------------------------------------------------------------*\
                         Class myincompressibleFluid Declaration
\*---------------------------------------------------------------------------*/

class myincompressibleFluid
:
    public incompressibleFluid
{
protected:

    // Diagnostic output

        fileName continuityOutput_;
        label continuityCapture_;


    // Protected Member Functions

        //- Calculate and print continuity statistics for a correction stage
        void writeCorrectPhiContinuity
        (
            const word& stage,
            const scalar threshold,
            const fileName& outputPath = fileName::null
        ) const;


        //- Write a residual with an independent copy of its capture-time mesh
        void writeCorrectPhiResidual
        (
            const volInternalScalarField& residual,
            const fileName& outputPath
        ) const;


public:

    //- Runtime type information
    TypeName("myincompressibleFluid");


    // Constructors

        //- Construct from region mesh
        explicit myincompressibleFluid(fvMesh& mesh);


    // Member Functions

        //- Correct flux after mesh changes, with optional diagnostics
        virtual void motionCorrector();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solvers
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
