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
#include "OFstream.H"
#include "OSspecific.H"
#include "fileOperation.H"

// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //

void Foam::solvers::myincompressibleFluid::writeCorrectPhiResidual
(
    const volInternalScalarField& residual,
    const fileName& outputPath
) const
{
    if (!mesh.conformal())
    {
        FatalErrorInFunction
            << "Continuity snapshots require a conformal mesh"
            << exit(FatalError);
    }

    dictionary controls;
    controls.add("application", word("foamRun"));
    controls.add("startFrom", word("startTime"));
    controls.add("startTime", runTime.value());
    controls.add("endTime", runTime.value());
    controls.add("deltaT", runTime.deltaTValue());
    controls.add("writeControl", word("timeStep"));
    controls.add("writeInterval", 1);
    controls.add("writeFormat", word("binary"));

    const fileName caseName = Pstream::parRun()
      ? fileName("processor" + Foam::name(Pstream::myProcNo())) : fileName(".");
    bool ok = true;
    if (Pstream::master())
    {
        mkDir(outputPath/"system");
        OFstream stream(outputPath/"system"/"controlDict");
        stream << "FoamFile { format ascii; class dictionary; "
            << "object controlDict; }" << nl;
        controls.write(stream, false);
        ok = stream.good();
    }
    reduce(ok, andOp());
    if (!ok)
    {
        FatalErrorInFunction
            << "Failed to write snapshot controls " << outputPath
            << exit(FatalError);
    }

    ok = mkDir(outputPath/caseName);
    reduce(ok, andOp());
    if (!ok)
    {
        FatalErrorInFunction
            << "Failed to create snapshot case " << outputPath/caseName
            << exit(FatalError);
    }

    Time outputTime(Time::controlDictName, outputPath, caseName, false);
    outputTime.setTime(runTime.value(), runTime.timeIndex());

    fvMesh outputMesh
    (
        IOobject
        (
            mesh.name(), outputTime.constant(), outputTime,
            IOobject::NO_READ, IOobject::NO_WRITE
        ),
        pointField(mesh.points()),
        faceList(mesh.faces()),
        labelList(mesh.faceOwner()),
        labelList(mesh.faceNeighbour())
    );

    List<polyPatch*> patches(mesh.boundary().size());
    wordList fieldTypes(patches.size(), "zeroGradient");
    forAll(patches, patchi)
    {
        const polyPatch& patch = mesh.poly().boundary()[patchi];
        const word& type = patch.type();
        if
        (
            type != "patch" && type != "wall" && type != "empty"
         && type != "wedge" && type != "symmetry" && type != "symmetryPlane"
         && type != "processor" && type != "processorCyclic"
         && type != "cyclic"
        )
        {
            FatalErrorInFunction
                << "Unsupported snapshot patch type " << type
                << exit(FatalError);
        }
        patches[patchi] = patch.clone(outputMesh.poly().boundary()).ptr();
        if
        (
            patch.coupled() || type == "empty" || type == "wedge"
         || type == "symmetry" || type == "symmetryPlane"
        )
        {
            fieldTypes[patchi] = type;
        }
    }
    outputMesh.addFvPatches(patches);

    List<pointZone*> pointZones(mesh.pointZones().size());
    List<faceZone*> faceZones(mesh.faceZones().size());
    List<cellZone*> cellZones(mesh.cellZones().size());
    forAll(pointZones, zonei)
    {
        pointZones[zonei] =
            mesh.pointZones()[zonei].clone(outputMesh.pointZones()).ptr();
    }
    forAll(faceZones, zonei)
    {
        faceZones[zonei] =
            mesh.faceZones()[zonei].clone(outputMesh.faceZones()).ptr();
    }
    forAll(cellZones, zonei)
    {
        cellZones[zonei] =
            mesh.cellZones()[zonei].clone(outputMesh.cellZones()).ptr();
    }
    outputMesh.addZones(pointZones, faceZones, cellZones);

    volScalarField outputResidual
    (
        IOobject
        (
            "continuityResidual", outputTime.name(), outputMesh,
            IOobject::NO_READ, IOobject::NO_WRITE, false
        ),
        outputMesh,
        dimensionedScalar(residual.dimensions(), 0),
        fieldTypes
    );
    outputResidual.primitiveFieldRef() = residual.primitiveField();
    outputResidual.correctBoundaryConditions();

    ok = outputMesh.writeMesh();
    ok = outputResidual.write() && ok;
    fileHandler().flush();
    reduce(ok, andOp());
    if (!ok)
    {
        FatalErrorInFunction
            << "Failed to write continuity snapshot " << outputPath
            << exit(FatalError);
    }

    if (Pstream::master())
    {
        OFstream marker(outputPath/"continuity.foam");
        marker << "continuityResidual " << outputTime.name() << endl;
        if (!marker.good())
        {
            FatalErrorInFunction
                << "Failed to complete continuity snapshot " << outputPath
                << exit(FatalError);
        }
    }
}


// ************************************************************************* //
