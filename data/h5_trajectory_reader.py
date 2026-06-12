#!/usr/bin/env python3
"""
h5_trajectory_reader.py
======================
Utility for reading and querying the consolidated HDF5 trajectory file.

Usage examples:
  # List all participants and sessions
  reader = H5TrajectoryReader()
  print(reader.list_all())

  # Read a specific trajectory
  df = reader.get_trajectory("Px10", "Apraxia_Imitation")

  # Read all sessions for a participant
  dfs = reader.get_participant_trajectories("Px10")

  # Read all trajectories matching a session type
  dfs = reader.get_session_type("HOI")

  # Get file statistics
  stats = reader.get_stats()

  # Close the file
  reader.close()

Command-line usage:
  python h5_trajectory_reader.py --list
  python h5_trajectory_reader.py --participant Px10
  python h5_trajectory_reader.py --participant Px10 --session Apraxia_Imitation
  python h5_trajectory_reader.py --session-type HOI
  python h5_trajectory_reader.py --stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import pandas as pd

_DEFAULT_HDF5 = (
    r"C:\Users\RL000009\OneDrive - Vrije Universiteit Brussel"
    r"\A-Skills\ASKILLS\Research stay ICL\data_apraxia\all_trajectories_synced.h5"
)


class H5TrajectoryReader:
    """Reader for consolidated trajectory HDF5 file."""

    def __init__(self, h5_path: Path | str | None = None):
        """
        Initialize reader.

        Args:
            h5_path: Path to HDF5 file. If None, uses default.
        """
        if h5_path is None:
            h5_path = _DEFAULT_HDF5

        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self.h5_file = h5py.File(self.h5_path, "r")

    def close(self) -> None:
        """Close the HDF5 file."""
        if self.h5_file:
            self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def list_participants(self) -> list[str]:
        """Get sorted list of all participants."""
        return sorted(self.h5_file.keys())

    def list_sessions(self, participant: str) -> list[str]:
        """Get sorted list of sessions for a participant."""
        if participant not in self.h5_file:
            raise ValueError(f"Participant not found: {participant}")
        return sorted(self.h5_file[participant].keys())

    def list_all(self) -> dict[str, list[str]]:
        """Get all participants and their sessions."""
        result = {}
        for px in self.list_participants():
            result[px] = self.list_sessions(px)
        return result

    def get_trajectory(
        self,
        participant: str,
        session: str,
        as_dict: bool = False,
    ) -> pd.DataFrame | dict:
        """
        Read a specific trajectory.

        Args:
            participant: Participant ID (e.g., "Px10")
            session: Session name (e.g., "Apraxia_Imitation")
            as_dict: If True, return as dict; else return as DataFrame

        Returns:
            DataFrame or dict with column names restored
        """
        if participant not in self.h5_file:
            raise ValueError(f"Participant not found: {participant}")

        group = self.h5_file[participant]
        if session not in group:
            raise ValueError(f"Session not found: {participant}/{session}")

        dset = group[session]
        data = dset[:]
        columns = [c.decode() if isinstance(c, bytes) else c for c in dset.attrs["columns"]]

        df = pd.DataFrame(data, columns=columns)

        return df if not as_dict else df.to_dict()

    def get_participant_trajectories(self, participant: str) -> dict[str, pd.DataFrame]:
        """Read all trajectories for a participant."""
        if participant not in self.h5_file:
            raise ValueError(f"Participant not found: {participant}")

        result = {}
        for session in self.list_sessions(participant):
            result[session] = self.get_trajectory(participant, session)

        return result

    def get_session_type(self, session_pattern: str) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Read all trajectories matching a session type (substring match).

        Args:
            session_pattern: Pattern to match in session names (e.g., "HOI", "Apraxia")

        Returns:
            Dict of {participant: {session: DataFrame}}
        """
        result = {}
        for px in self.list_participants():
            sessions = self.list_sessions(px)
            matching = [s for s in sessions if session_pattern in s]
            if matching:
                result[px] = {}
                for session in matching:
                    result[px][session] = self.get_trajectory(px, session)

        return result

    def get_stats(self) -> dict:
        """Get file statistics."""
        file_size_mb = self.h5_path.stat().st_size / (1024 * 1024)

        participants = self.list_participants()
        total_rows = 0
        total_cols = 0
        session_count = 0

        for px in participants:
            for session in self.list_sessions(px):
                dset = self.h5_file[px][session]
                total_rows += dset.shape[0]
                total_cols = dset.shape[1]  # Should be same for all
                session_count += 1

        return {
            "h5_file": str(self.h5_path),
            "file_size_mb": round(file_size_mb, 2),
            "num_participants": len(participants),
            "num_sessions": session_count,
            "total_rows": total_rows,
            "columns_per_row": total_cols,
            "participants": participants,
        }

    def print_summary(self) -> None:
        """Print a summary of the HDF5 file."""
        stats = self.get_stats()
        print(f"\nHDF5 Trajectory File Summary")
        print(f"{'='*60}")
        print(f"File: {stats['h5_file']}")
        print(f"Size: {stats['file_size_mb']} MB")
        print(f"Participants: {stats['num_participants']}")
        print(f"Sessions: {stats['num_sessions']}")
        print(f"Total rows: {stats['total_rows']:,}")
        print(f"Columns per trajectory: {stats['columns_per_row']}")
        print(f"\nParticipants: {', '.join(stats['participants'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read consolidated trajectory HDF5 file")
    parser.add_argument(
        "--h5-file",
        default=_DEFAULT_HDF5,
        help="Path to HDF5 file",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all participants and sessions",
    )
    parser.add_argument(
        "--participant",
        help="Get sessions for a participant",
    )
    parser.add_argument(
        "--session",
        help="Get a specific trajectory (requires --participant)",
    )
    parser.add_argument(
        "--session-type",
        help="Get all trajectories matching a session type (e.g., 'HOI', 'Apraxia')",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print file statistics",
    )
    parser.add_argument(
        "--export",
        help="Export a trajectory to CSV (format: participant/session)",
    )
    args = parser.parse_args()

    reader = H5TrajectoryReader(args.h5_file)

    try:
        if args.list:
            all_data = reader.list_all()
            for px in sorted(all_data.keys()):
                print(f"\n{px}:")
                for session in all_data[px]:
                    print(f"  - {session}")

        elif args.stats:
            reader.print_summary()

        elif args.participant and args.session:
            df = reader.get_trajectory(args.participant, args.session)
            print(f"\n{args.participant} / {args.session}")
            print(f"Shape: {df.shape}")
            print(f"\nFirst 5 rows:")
            print(df.head())

        elif args.participant:
            sessions = reader.list_sessions(args.participant)
            print(f"\n{args.participant} sessions ({len(sessions)} total):")
            for session in sessions:
                df = reader.get_trajectory(args.participant, session)
                print(f"  {session:30s} {df.shape[0]:>8,} rows x {df.shape[1]:>3} cols")

        elif args.session_type:
            trajectories = reader.get_session_type(args.session_type)
            total_rows = sum(
                sum(df.shape[0] for df in sessions.values())
                for sessions in trajectories.values()
            )
            print(f"\nTrajectories matching '{args.session_type}' ({len(trajectories)} participants):")
            for px in sorted(trajectories.keys()):
                for session, df in sorted(trajectories[px].items()):
                    print(f"  {px:5s} {session:30s} {df.shape[0]:>8,} rows")
            print(f"\nTotal rows: {total_rows:,}")

        elif args.export:
            parts = args.export.split("/")
            if len(parts) != 2:
                print("Error: --export format should be 'participant/session'")
                return
            px, session = parts
            df = reader.get_trajectory(px, session)
            output_path = Path(f"{px}_{session}_trajectories_synced.csv")
            df.to_csv(output_path, index=False)
            print(f"Exported: {output_path}")

        else:
            reader.print_summary()

    finally:
        reader.close()


if __name__ == "__main__":
    main()
