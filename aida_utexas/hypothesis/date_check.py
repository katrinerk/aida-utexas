"""
Author: Katrin Erk, May 2019
- Check temporal constraints

Update: Pengxiang Cheng, Aug 2020
- Refactoring and clean-up
- Needs further rewriting to accommodate to new M36 temporal specifications
"""

import datetime
import logging
from typing import Dict, Optional

from aida_utexas.aif.json_graph import ERENode


# AIDA temporal info handling,
# with missing entries for days/months/years
class AidaIncompleteDate:
    def __init__(self, year, month, day):
        self.year: Optional[int] = self._check_date_str(year, 'year')
        self.month: Optional[int] = self._check_date_str(month, 'month')
        self.day: Optional[int] = self._check_date_str(day, 'day')

    def to_json(self):
        return {
            'year': self.year,
            'month': self.month,
            'day': self.day
        }

    @classmethod
    def from_json(cls, date_json: Dict):
        return cls(
            year=date_json.get('year', None),
            month=date_json.get('month', None),
            day=date_json.get('day', None)
        )

    @staticmethod
    def _check_date_str(date, date_field):
        if date is not None and not isinstance(date, int):
            if isinstance(date, str) and date.isdigit():
                date = int(date)
            else:
                if not (isinstance(date, str) and (date.strip("X") == "")):
                    logging.warning(f'unexpected type for {date_field}: {date}, {type(date)}')
                date = None
        return date

    # convert to python datetime.date object
    def to_python_date(self):
        year = self.year if self.year is not None else 1
        month = self.month if self.month is not None else 1
        day = self.day if self.day is not None else 1

        return datetime.date(year, month, day)

    # whether the date is the last day of a month
    def is_last_day_of_month(self):
        if self.day is None:
            return False

        curr_date = self.to_python_date()
        next_date = curr_date + datetime.timedelta(1)
        return curr_date.month != next_date.month

    # add a day to the date
    def add_day(self):
        # if day is not set, return the same date
        if self.day is None:
            return AidaIncompleteDate(self.year, self.month, self.day)

        next_date = self.to_python_date() + datetime.timedelta(1)

        new_year = next_date.year if self.year is not None else None
        new_month = next_date.month if self.month is not None else None

        return AidaIncompleteDate(new_year, new_month, next_date.day)

    # can t1 be before t2?
    def is_before(self, date2, add_a_day: bool = False):
        if add_a_day:
            date2 = date2.add_day()

        if self.year is None or date2.year is None:
            # we don't know which year one of the dates given is, so t1 can always be before t2
            return True

        elif self.year > date2.year:
            # t1 is in a later year, so not before t2
            return False

        elif self.year < date2.year:
            # t1 is in an earlier year than t2, so definitely before t2
            return True

        # same year, move on to month
        if self.month is None and date2.month is None:
            # we don't know the month in either, so t1 can always be before t2
            return True

        elif self.month is None and date2.month > 1:
            # we dont know t1's month, and t2 is in Feb at the earliest
            return True

        elif date2.month is None and self.month < 12:
            # we don't know t2's month, and t1 is November at the latest
            return True

        elif self.month is not None and date2.month is not None and self.month < date2.month:
            # t1 earlier month than t2, so definitely before
            return True

        elif self.month is not None and date2.month is not None and self.month > date2.month:
            # t1 later month than t2, so definitely not before
            return False

        # we need to compare days
        if self.day is None and date2.day is None:
            # we don't know the day in either, so t1 can always be before t2
            return True

        if self.day is not None and date2.day is not None and self.day < date2.day:
            # t1 earlier day than t2, so definitely before
            return True

        elif date2.day is not None and self.day is None and date2.day > 2:
            # t1, t2 can be same year and month. if t2.day is at least 2, t1 can always be before t2
            return True

        elif self.day is not None and date2.day is None:
            # if t1 is the last day of a month, then t1 cannot be before t2
            if self.is_last_day_of_month():
                return False
            else:
                return True

        # comparison of days showed that t1 cannot be before t2
        return False

    # can t1 be the same as t2?
    def is_eq(self, date2, add_a_day: bool = False):
        if add_a_day:
            # try adding a day either way
            next_date2 = date2.add_day()
            if self.is_eq(next_date2):
                return True
            next_date = self.add_day()
            if next_date.is_eq(date2):
                return True

        # try exact match of the year
        if self.year is not None and date2.year is not None and self.year != date2.year:
            return False

        # at this point we know the year can be compatible, try exact match of the month
        if self.month is not None and date2.month is not None and self.month != date2.month:
            return False

        # at this point we know the month can be compatible, try exact match of the day
        if self.day is not None and date2.day is not None and self.day != date2.day:
            return False

        # at this point, the whole dates can be compatible
        return True


# returns true if the ERE has no temporal constraint or matches the temporal constraint
# leeway levels: 0 = no leeway, 1 = one day leeway, 2 = don't test
def temporal_constraint_match(ere_node: ERENode, constraint_entry: Dict[str, AidaIncompleteDate],
                              leeway: int = 0):
    if leeway > 1:
        # no testing, assume it fits.
        return True

    if constraint_entry is None:
        # no temporal constraint, no problem
        return True

    # does it have a temporal entry?
    if not ere_node.ldcTime:
        # we were looking for a temporal entry but did not find one.
        # this is a constraint violation.
        return False

    # we have a match if at least one of the temporal statements of the entry for qvar_filler match
    for ldc_time in ere_node.ldcTime:
        # if there is no start time, we add a None place holder
        for start_time in ldc_time.get('start', [None]):
            # if there is no end time, we also add a None place holder
            for end_time in ldc_time.get('end', [None]):
                # we can only match the temporal constraint with at least one of start / end time.
                if start_time is None and end_time is None:
                    continue
                # this time matched the temporal constraint
                if temporal_constraint_match_pair(start_time, end_time, constraint_entry, leeway):
                    return True

    # we found no time for our event that matched the temporal constraint
    return False


def temporal_constraint_match_pair(start_time: Dict, end_time: Dict,
                                   constraint_entry: Dict[str, AidaIncompleteDate],
                                   leeway: int = 0):
    start_time = possibly_flatten_ldc_time(start_time)
    end_time = possibly_flatten_ldc_time(end_time)

    # construct AidaIncompleteDate from start_time and end_time
    start_date = AidaIncompleteDate.from_json(start_time) if start_time is not None else None
    end_date = AidaIncompleteDate.from_json(end_time) if end_time is not None else None

    # start_time and end_time have a timeType of ON/BEFORE/AFTER.
    start_time_type, end_time_type = start_time['timeType'], end_time['timeType']

    # whether to allow add_a_day leeway
    add_a_day = leeway > 0

    # constraint_entry has 'start_time' and 'end_time', no time type.
    if 'start_time' in constraint_entry and start_date is not None:
        # we can compare start times. otherwise, assume compatibility.

        if start_time_type == 'BEFORE' and not constraint_entry['start_time'].is_before(
                start_date, add_a_day):
            # event start before t and qt.start_time not before t: constraint violation
            return False

        if start_time_type == 'AFTER' and not start_date.is_before(
                constraint_entry['start_time'], add_a_day):
            # event start after t and qt.start_time before t: constraint violation
            return False

        if start_time_type == 'ON' and not start_date.is_eq(
                constraint_entry['start_time'], add_a_day):
            # event start and qt.start_time not equal
            return False

    if 'end_time' in constraint_entry and end_date is not None:
        # we can compare end times. otherwise, assume compatibility

        if end_time_type == 'BEFORE' and not constraint_entry['end_time'].is_before(
                end_date, add_a_day):
            # event end before t and t before qt.end_time: constraint violation
            return False
        if end_time_type == 'AFTER' and not end_date.is_before(
                constraint_entry['end_time'], add_a_day):
            # event end after t and qt.end_time before t: constraint violation
            return False
        if end_time_type == 'ON' and not end_date.is_eq(constraint_entry['end_time'], add_a_day):
            # event end and qt.end_time not  equal
            return False

    return True


def possibly_flatten_ldc_time(ldc_time: Dict):
    if ldc_time is None:
        return None
    return {key: val[0] if isinstance(val, list) else val for key, val in ldc_time.items()}
