import { NextRequest, NextResponse } from 'next/server';
import { readFile } from 'fs/promises';
import { join } from 'path';

export interface DatasetRecord {
  id: number;
  text: string;
  labels: 0 | 1;
  platform: 'Twitter' | 'Instagram';
  char_count: number;
  word_count: number;
}

export interface DatasetResponse {
  records: DatasetRecord[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const page = parseInt(searchParams.get('page') || '1');
    const limit = parseInt(searchParams.get('limit') || '20');
    const labelFilter = searchParams.get('label');
    const platformFilter = searchParams.get('platform');

    // Load CSV file from project root
    const csvPath = join(process.cwd(), '../../dataset/indonesian_hate_speech.csv');
    const csvContent = await readFile(csvPath, 'utf-8');

    // Parse CSV
    const lines = csvContent.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    // Find column indices
    const textIdx = headers.findIndex(h => h === 'text');
    const labelIdx = headers.findIndex(h => h === 'labels');
    const platformIdx = headers.findIndex(h => h === 'platform' || h === 'source');

    if (textIdx === -1 || labelIdx === -1) {
      return NextResponse.json(
        { error: 'Required columns not found in CSV' },
        { status: 500 }
      );
    }

    // Parse records
    const records: DatasetRecord[] = [];
    
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      // Handle CSV parsing with potential commas in text
      const parts: string[] = [];
      let current = '';
      let inQuotes = false;

      for (let j = 0; j < line.length; j++) {
        const char = line[j];
        
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          parts.push(current.trim().replace(/^"|"$/g, ''));
          current = '';
        } else {
          current += char;
        }
      }
      parts.push(current.trim().replace(/^"|"$/g, ''));

      if (parts.length < headers.length) continue;

      const text = parts[textIdx] || '';
      const label = parseInt(parts[labelIdx]) as 0 | 1;
      const platform = platformIdx >= 0 ? parts[platformIdx] : 'Twitter';

      // Apply filters
      if (labelFilter && label !== parseInt(labelFilter)) continue;
      if (platformFilter && platform.toLowerCase() !== platformFilter.toLowerCase()) continue;

      // Skip invalid records
      if (!text || (label !== 0 && label !== 1)) continue;

      const wordCount = text.split(/\s+/).filter(w => w.length > 0).length;
      
      records.push({
        id: i,
        text,
        labels: label,
        platform: platform as 'Twitter' | 'Instagram',
        char_count: text.length,
        word_count: wordCount
      });
    }

    // Pagination
    const total = records.length;
    const startIdx = (page - 1) * limit;
    const endIdx = startIdx + limit;
    const paginatedRecords = records.slice(startIdx, endIdx);
    const hasMore = endIdx < total;

    return NextResponse.json({
      records: paginatedRecords,
      total,
      page,
      pageSize: limit,
      hasMore
    });

  } catch (error) {
    console.error('Error loading dataset:', error);
    return NextResponse.json(
      { error: 'Failed to load dataset', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}