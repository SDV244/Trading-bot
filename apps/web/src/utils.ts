export function formatNumber(value: string | number | null | undefined, digits = 2): string {
  if (value === null || value === undefined) {
    return "-";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return String(value);
  }
  return numeric.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleString();
}

export function formatTime(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleTimeString();
}

