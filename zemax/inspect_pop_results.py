"""Run the bundled Basic Propagation sample and inspect POP result grids."""

from pathlib import Path

from zosapi_utils import ZOSApplication


def public_names(value):
    return sorted(name for name in dir(value) if not name.startswith("_"))


def main():
    with ZOSApplication() as zos:
        sample = Path(zos.application.SamplesDir) / "Physical Optics" / "Basic Propagation.zos"
        zos.system.LoadFile(str(sample), False)
        pop = zos.system.Analyses.New_Analysis(
            zos.ZOSAPI.Analysis.AnalysisIDM.PhysicalOpticsPropagation
        )
        pop.ApplyAndWaitForCompletion()
        results = pop.GetResults()
        print(f"status={pop.StatusMessages}")
        print("results_members=" + ",".join(public_names(results)))
        print(f"grids={results.NumberOfDataGrids};series={results.NumberOfDataSeries}")
        grid = results.GetDataGrid(0)
        print("grid_members=" + ",".join(public_names(grid)))
        print(
            f"shape={grid.Values.GetLength(0)}x{grid.Values.GetLength(1)};"
            f"min=({grid.MinX},{grid.MinY});step=({grid.Dx},{grid.Dy})"
        )
        pop.Close()


if __name__ == "__main__":
    main()
