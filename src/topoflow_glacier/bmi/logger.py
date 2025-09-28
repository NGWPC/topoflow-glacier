"""Logging module to integrate with NGEN-CERF"""

from __future__ import annotations

import getpass
import logging
import os
import sys
import time
from datetime import UTC, datetime

logger = logging.getLogger()
_configured = False

MODULE_NAME = "Topoflow-Glacier"
LOG_DIR_NGENCERF = "/ngencerf/data"  # ngenCERF log directory string if environement var empty.
LOG_DIR_DEFAULT = "run-logs"  # Default parent log directory string if env var empty  & ngencerf dosn't exist
LOG_FILE_EXT = "log"  # Log file name extension
DS = "/"  # Directory separator
LOG_MODULE_NAME_LEN = 8  # Width of module name for log entries

EV_EWTS_LOGGING = "NGEN_EWTS_LOGGING"  # Enable/disable of Error Warning and Trapping System
EV_NGEN_LOGFILEPATH = "NGEN_LOG_FILE_PATH"  # ngen log file
EV_MODULE_LOGLEVEL = "TOPOFLOW_GLACIER_LOGLEVEL"  # This modules log level
EV_MODULE_LOGFILEPATH = "TOPOFLOW_GLACIER_LOGFILEPATH"  # This modules log full log filename


class CustomFormatter(logging.Formatter):
    """A custom formatting class for logging"""

    LEVEL_NAME_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "SEVERE",
        logging.CRITICAL: "FATAL",
    }

    def format(self, record):
        """Formats the file name"""
        original_levelname = record.levelname
        record.levelname = self.LEVEL_NAME_MAP.get(record.levelno, original_levelname)
        record.levelname_padded = record.levelname.ljust(7)[:7]  # Exactly 7 chars
        formatted = super().format(record)
        record.levelname = original_levelname  # Restore original in case it's reused
        return formatted


def create_timestamp(date_only: bool = False, iso: bool = False, append_ms: bool = False) -> str:
    """Created the timestamp for the log file"""
    now = datetime.now(UTC)

    if date_only:
        ts_base = now.strftime("%Y%m%d")
    elif iso:
        ts_base = now.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        ts_base = now.strftime("%Y%m%dT%H%M%S")

    if append_ms:
        ms_str = f".{now.microsecond // 1000:03d}"
        return ts_base + ms_str
    else:
        return ts_base


def get_log_file_path():
    """Returns the file path of the module logs"""
    appendEntries = True
    moduleLogEnvExists = False
    logFileDir = ""
    moduleEnvVar = os.getenv(EV_MODULE_LOGFILEPATH, "")
    if moduleEnvVar:
        logFilePath = moduleEnvVar
        moduleLogEnvExists = True
    else:
        ngenEnvVar = os.getenv(EV_NGEN_LOGFILEPATH, "")
        if ngenEnvVar:
            logFilePath = ngenEnvVar
        else:
            appendEntries = False
            if os.path.isdir(LOG_DIR_NGENCERF):
                logFileDir = LOG_DIR_NGENCERF + DS + LOG_DIR_DEFAULT
            else:
                logFileDir = os.path.expanduser("~") + DS + LOG_DIR_DEFAULT
            try:
                os.makedirs(logFileDir, exist_ok=True)
                # Set full log path
                username = getpass.getuser()
                if username:
                    logFileDir = logFileDir + DS + username
                else:
                    logFileDir = logFileDir + DS + create_timestamp(True)
                # Create directory
                with os.makedirs(logFileDir, exist_ok=True):
                    logFilePath = (
                        logFileDir + DS + MODULE_NAME + "_" + create_timestamp() + "." + LOG_FILE_EXT
                    )
            except TypeError:
                logFilePath = ""

    # Ensure log file can be opened and set module env var
    try:
        if logFilePath:
            if appendEntries:
                logFile = open(logFilePath, "a")
            else:
                logFile = open(logFilePath, "w")  # noqa: F841
            if not moduleLogEnvExists:
                os.environ[EV_MODULE_LOGFILEPATH] = logFilePath
                print(f"Module {MODULE_NAME} Log File: {logFilePath}", flush=True)
        else:
            raise OSError
    except:
        print(f"Unable to open log file for {MODULE_NAME}: {logFilePath}", flush=True)
        print("Log entries will be writen to stdout", flush=True)

    return logFilePath, appendEntries


def get_log_level() -> str:
    """Returns the module log level"""
    levelEnvVar = os.getenv(EV_MODULE_LOGLEVEL, "")
    if levelEnvVar:
        print(f"{EV_MODULE_LOGLEVEL}={levelEnvVar}", flush=True)
        return levelEnvVar.strip().upper()
    else:
        print(f"{EV_MODULE_LOGLEVEL} not found. Using INFO log level", flush=True)
        return "INFO"


def translate_ngwpc_log_level(ngwpc_log_level: str) -> str:
    """Translates a log level to verbiage used by NGWPC"""
    ll = ngwpc_log_level.strip().upper()
    if ll == "SEVERE":
        return "ERROR"
    elif ll == "FATAL":
        return "CRITICAL"
    return ll


def configure_logging():
    """
    Set logging level and specify logger configuration based on environment variables set by ngen

    Arguments
    ---------
    logging._Level: Log level
    ** Not used in NGWPC version. Instead the ngen logger defines environment variables that are read

    Returns
    -------
    None

    Notes
    -----
    In the absense of logging environment variables the log level defaults to INFO and
    the pathname is set as follows:
        - Use the module log file if available (unset when first run by ngen), otherwise
        - Use ngen log file if available, otherwise
        - Use /ngencerf/data/run-logs/<username>/<module>_<YYMMDDTHHMMSS> if available, otherwise
        - Use ~/run-logs/<YYYYMMDD>/<module>_<YYMMDDTHHMMSS>
        - Onced opened, save the full log path to the modules log environment variable so
          it is only opened once for each ngen run (vs for each catchment)

    See also https://docs.python.org/3/library/logging.html

    """
    global _configured  # Tell Python this refers to the module-level variable
    modulePathEnvVarSet = os.getenv(EV_MODULE_LOGFILEPATH, "")
    if modulePathEnvVarSet and _configured:
        return  # Nothing to do — already configured, and env var is set
    elif not modulePathEnvVarSet and _configured:
        # Need set a log file since the ngen.log was truncated.
        logFilePath, appendEntries = get_log_file_path()
        if logFilePath:
            # Set the open mode
            openMode = "a" if appendEntries else "w"
            handler = logging.FileHandler(logFilePath, mode=openMode)
        else:
            handler = logging.StreamHandler(sys.stdout)
        return

    loggingEnabled = True
    moduleEnvVar = os.getenv(EV_EWTS_LOGGING, "")
    if moduleEnvVar:
        print(f"{EV_EWTS_LOGGING}={moduleEnvVar}", flush=True)
        if moduleEnvVar == "DISABLED":
            loggingEnabled = False
    else:
        print(f"{EV_EWTS_LOGGING} not found.", flush=True)

    if not loggingEnabled:
        print(f"Module {MODULE_NAME} Logging DISABLED", flush=True)
        logging.disable(logging.CRITICAL)  # Disables all logs at CRITICAL and below (i.e., everything)
    else:
        print(f"Module {MODULE_NAME} Logging ENABLED", flush=True)

        # Get the log file name from env var or a default
        logFilePath, appendEntries = get_log_file_path()
        if logFilePath:
            # Set the open mode
            openMode = "a" if appendEntries else "w"
            handler = logging.FileHandler(logFilePath, mode=openMode)
        else:
            handler = logging.StreamHandler(sys.stdout)

        # Get the log level from env var or a default
        log_level = get_log_level()

        # Format the module name: uppercase, fixed length, left-justify or trimmed
        formatted_module = MODULE_NAME.upper().ljust(LOG_MODULE_NAME_LEN)[:LOG_MODULE_NAME_LEN]

        # Apply custom formatter
        formatted_module = MODULE_NAME.upper().ljust(LOG_MODULE_NAME_LEN)[:LOG_MODULE_NAME_LEN]
        formatter = CustomFormatter(
            fmt=f"%(asctime)s.%(msecs)03d {formatted_module} %(levelname_padded)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)

        # Setup root logger
        logging.getLogger().handlers.clear()  # Clear any default handlers
        logging.getLogger().setLevel(translate_ngwpc_log_level(log_level))
        logging.getLogger().addHandler(handler)

        # Ensure UTC timestamps
        logging.Formatter.converter = time.gmtime

        # Save the current log level
        current_level = logging.getLogger().getEffectiveLevel()

        try:
            # Temporarily set log level to INFO
            logging.getLogger().setLevel(logging.INFO)

            # Log the message at INFO level
            logging.info(f"Log level set to {log_level}")
            print(f"Module {MODULE_NAME} Log Level set to {log_level}", flush=True)
        finally:
            # Restore the original log level
            logging.getLogger().setLevel(current_level)

    # Set this true so the logger is only configured once
    _configured = True
